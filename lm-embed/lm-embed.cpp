#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

typedef float Float;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf Vector;
typedef Eigen::ArrayXf Array;
typedef Eigen::SparseMatrix<Float> SparseMatrix;

typedef unsigned int uint;
typedef uint VocID;

typedef std::map<std::string,VocID> IDMap;
typedef std::pair<uint,VocID> VocCountPair;

const Float TINY = Float(1e-30);
const Float ZERO = 0;

struct Trigram {
	VocID prev;
	VocID cur;
	VocID next;

	bool operator==(const Trigram &o) const {
		return o.prev == prev && o.cur == cur && o.next == next;
	}
};

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

struct TrigramHash {
	std::size_t operator()(const Trigram &o) const {
		std::size_t seed = 0;
		hash_combine(seed, o.prev);
		hash_combine(seed, o.cur);
		hash_combine(seed, o.next);
		return seed;
	}
};

typedef std::unordered_map<Trigram,uint,TrigramHash> TrigramCountMap;

const uint PREDICTION_VOCSIZE = 1000;

TrigramCountMap load_trigrams(uint &vocsize);

template<class Derived> void softmax(const Eigen::MatrixBase<Derived> &m);

TrigramCountMap load_trigrams(uint &next_vocid) {
	IDMap idmap;
	std::vector<VocID> corpus;
	std::vector<VocCountPair> counts;
	TrigramCountMap tgmap;

	next_vocid = 1;
	std::string line;
	while(getline(std::cin, line)) {
		std::istringstream is(line);
		std::string word;
		while(is >> word) {
			IDMap::const_iterator it = idmap.find(word);
			if(it != idmap.end()) {
				corpus.push_back(it->second);
				counts[it->second].first++;
			} else {
				corpus.push_back(next_vocid);
				counts.push_back(std::make_pair(1, next_vocid));
				idmap.insert(std::make_pair(word, next_vocid));
				next_vocid++;
			}
		}
		corpus.push_back(0);
		counts[0].first++;
	}

	std::sort(counts.begin(), counts.end(), std::greater<VocCountPair>());
	std::vector<VocID> idtrans(counts.size());
	for(uint i = 0; i < counts.size(); i++)
		idtrans[counts[i].second] = i;
	counts.clear();

	std::ofstream vocstr("vocabulary");
	for(IDMap::const_iterator it = idmap.begin(); it != idmap.end(); ++it)
		vocstr << it->first << '\t' << idtrans[it->second] << '\n';
	idmap.clear();

	VocID prev = idtrans[0];
	VocID cur = idtrans[corpus.front()];
	for(uint i = 1; i < corpus.size(); i++) {
		VocID next = idtrans[corpus[i]];

		Trigram tg;
		tg.prev = std::min(prev, PREDICTION_VOCSIZE);
		tg.cur = cur;
		tg.next = std::min(next, PREDICTION_VOCSIZE);

		TrigramCountMap::iterator it = tgmap.find(tg);
		if(it != tgmap.end())
			it->second++;
		else
			tgmap.insert(std::make_pair(tg, 1));

		prev = cur;
		cur = next;
	}

	return tgmap;
}

struct Batch {
	uint nitems;
	SparseMatrix input;
	Matrix targets;
};

class TrigramSampler {
private:
	mutable std::mt19937 generator_;

	uint inputvocsize_;
	std::vector<Trigram> trigrams_;
	std::vector<uint> counts_;

public:
	TrigramSampler(uint ivoc, const TrigramCountMap &tgmap);

	Batch extract(uint size);
	Batch operator()(uint size, std::vector<uint> *sample = NULL) const;
};

TrigramSampler::TrigramSampler(uint vocsize, const TrigramCountMap &tgmap) :
		inputvocsize_(vocsize) {
	trigrams_.reserve(tgmap.size());
	counts_.reserve(tgmap.size() + 1);
	uint last_count = 0;
	counts_.push_back(0);
	for(TrigramCountMap::const_iterator it = tgmap.begin(); it != tgmap.end(); ++it) {
		trigrams_.push_back(it->first);
		last_count += it->second;
		counts_.push_back(last_count);
	}
}

Batch TrigramSampler::extract(uint size) {
	std::vector<uint> sample;
	sample.reserve(size);
	Batch b = operator()(size, &sample);
	std::sort(sample.begin(), sample.end());
	for(uint i = 0, j = 0, diff = 0; i < counts_.size(); i++) {
		while(j < sample.size() && i == sample[j]) {
			j++;
			if(i > 0 && counts_[i] - diff > counts_[i - 1] ||
					i == 0 && counts_[i] > diff)
				diff++;
		}
		counts_[i] -= diff;
	}
	return b;
}

Batch TrigramSampler::operator()(uint size, std::vector<uint> *sample) const {
	typedef Eigen::Triplet<Float> Trip;
	std::vector<Trip> itrip; //, ttrip;

	itrip.reserve(size);
	//ttrip.reserve(2 * size);

	Batch batch;
	batch.nitems = size;
	batch.targets.resize(size, 2 * (PREDICTION_VOCSIZE + 1));

	std::uniform_int_distribution<uint> dist(0, counts_.back());
	for(uint i = 0; i < size; i++) {
		uint s = dist(generator_);
		if(sample)
			sample->push_back(s);
		uint idx = std::upper_bound(counts_.begin(), counts_.end(), s) - counts_.begin() - 1;
		itrip.push_back(Trip(i, trigrams_[idx].cur, 1));
		batch.targets(i, trigrams_[idx].prev) = 1;
		batch.targets(i, trigrams_[idx].next + PREDICTION_VOCSIZE + 1) = 1;
		//ttrip.push_back(Trip(i, trigrams_[idx].prev, 1));
		//ttrip.push_back(Trip(i, trigrams_[idx].next + PREDICTION_VOCSIZE + 1, 1));
	}

	batch.input.resize(size, inputvocsize_);
	batch.input.setFromTriplets(itrip.begin(), itrip.end());
	//batch.targets.setFromTriplets(ttrip.begin(), ttrip.end());

	return batch;
}

template<class Config> class LMEmbedCoefficients;
template<class Config> class LMEmbed;

template<class Activation>
struct LMEmbedConfiguration {
	typedef LMEmbedConfiguration<Activation> Config;

	typedef Activation ActivationFunction;
	typedef LMEmbedCoefficients<Config> Coeff;
	typedef LMEmbed<Config> Network;

	uint ninput;
	uint nhid;
	uint redvocsize;

	ActivationFunction activation;

	LMEmbedConfiguration(ActivationFunction act, uint i, uint h, uint rv) :
		activation(act), ninput(i), nhid(h), redvocsize(rv) {}
};

struct SigmoidActivation {
	template<class Derived>
	Matrix f(const Eigen::MatrixBase<Derived> &x) {
		return (Float(1) + (-x.array()).exp()).inverse().matrix();
	}

	template<class Derived>
	Matrix df(const Eigen::MatrixBase<Derived> &x, const Eigen::MatrixBase<Derived> &y) {
		return (y.array() * (Float(1) - y.array())).matrix();
	}
};

template<class Configuration>
class LMEmbedCoefficients {
private:
	Array W_;

	Configuration conf_;

	uint s_inphid;
	uint s_hidbias;
	uint s_hidout;
	uint s_outbias;

	LMEmbedCoefficients(const Configuration &conf);

public:
	typedef Eigen::Map<Matrix,Eigen::Aligned> MatrixMap;
	typedef Eigen::Map<Vector,Eigen::Aligned> VectorMap;
	typedef Eigen::Map<const Matrix,Eigen::Aligned> ConstMatrixMap;
	typedef Eigen::Map<const Vector,Eigen::Aligned> ConstVectorMap;

	static LMEmbedCoefficients zeros(const Configuration &conf);
	static LMEmbedCoefficients ones(const Configuration &conf);
	static LMEmbedCoefficients init_weights(const Configuration &conf);

	Array &array() {
		return W_;
	}
	const Array &array() const {
		return W_;
	}

	MatrixMap inphid() {
		return MatrixMap(W_.data() + s_inphid, conf_.ninput, conf_.nhid);
	}
	const ConstMatrixMap inphid() const {
		return ConstMatrixMap(W_.data() + s_inphid, conf_.ninput, conf_.nhid);
	}

	VectorMap hidbias() {
		return VectorMap(W_.data() + s_hidbias, conf_.nhid);
	}
	const ConstVectorMap hidbias() const {
		return ConstVectorMap(W_.data() + s_hidbias, conf_.nhid);
	}

	MatrixMap hidout() {
		return MatrixMap(W_.data() + s_hidout, conf_.nhid, 2 * conf_.redvocsize);
	}
	const ConstMatrixMap hidout() const {
		return ConstMatrixMap(W_.data() + s_hidout, conf_.nhid, 2 * conf_.redvocsize);
	}

	VectorMap outbias() {
		return VectorMap(W_.data() + s_outbias, 2 * conf_.redvocsize);
	}
	const ConstVectorMap outbias() const {
		return ConstVectorMap(W_.data() + s_outbias, 2 * conf_.redvocsize);
	}
};

template<class Configuration>
LMEmbedCoefficients<Configuration>::LMEmbedCoefficients(const Configuration &conf) : conf_(conf) {
	const uint ALIGNCOUNT = 128 / sizeof(Float);

	uint nelems = 0;

	s_inphid = nelems;
	nelems += conf.ninput * conf.nhid;
	if(nelems % ALIGNCOUNT > 0)
		nelems += ALIGNCOUNT - nelems % ALIGNCOUNT;

	s_hidbias = nelems;
	nelems += conf.nhid;
	if(nelems % ALIGNCOUNT > 0)
		nelems += ALIGNCOUNT - nelems % ALIGNCOUNT;

	s_hidout = nelems;
	nelems += conf.nhid * 2 * conf.redvocsize;
	if(nelems % ALIGNCOUNT > 0)
		nelems += ALIGNCOUNT - nelems % ALIGNCOUNT;

	s_outbias = nelems;
	nelems += 2 * conf.redvocsize;

	W_.resize(nelems);
}

template<class Configuration>
LMEmbedCoefficients<Configuration> LMEmbedCoefficients<Configuration>::zeros(const Configuration &conf) {
	LMEmbedCoefficients coef(conf);
	coef.array().setZero();
	return coef;
}

template<class Configuration>
LMEmbedCoefficients<Configuration> LMEmbedCoefficients<Configuration>::ones(const Configuration &conf) {
	LMEmbedCoefficients coef(conf);
	coef.array().setOnes();
	return coef;
}

template<class Configuration>
LMEmbedCoefficients<Configuration> LMEmbedCoefficients<Configuration>::init_weights(const Configuration &conf) {
	LMEmbedCoefficients coef(conf);

	std::default_random_engine generator;

	Float amp_inphid = Float(4) * std::sqrt(Float(conf.ninput + conf.nhid));
	std::uniform_real_distribution<Float> dist_inphid(-amp_inphid, amp_inphid);
	std::generate(coef.inphid().data(), coef.inphid().data() + conf.ninput * conf.nhid,
		std::bind(dist_inphid, generator));

	Float amp_hidout = Float(4) * std::sqrt(Float(conf.nhid + 2 * conf.redvocsize));
	std::uniform_real_distribution<Float> dist_hidout(-amp_hidout, amp_hidout);
	std::generate(coef.hidout().data(), coef.hidout().data() + conf.nhid * 2 * conf.redvocsize,
		std::bind(dist_hidout, generator));

	coef.hidbias().setZero();
	coef.outbias().setZero();

	return coef;
}

template<class Configuration>
class LMEmbed {
private:
	Configuration conf_;

	Matrix input_;
	Matrix hidin_;
	Matrix hidden_;
	Matrix output_;

public:
	typedef typename Configuration::Coeff Coeff;

	LMEmbed(const Configuration &conf) : conf_(conf) {}

	const Matrix &fprop(const Matrix &input, const Coeff &W);
	const Coeff bprop(const Matrix &targets, const Coeff &W);
};

template<class Configuration>
const Matrix &LMEmbed<Configuration>::fprop(const Matrix &input, const Coeff &W) {
	hidin_.noalias() = input_ * W.inphid() + W.hidbias();
	hidden_.noalias() = conf_.activation.f(hidin_);
	output_.noalias() = hidden_ * W.hidout() + W.outbias();
	softmax(output_.leftCols(conf_.redvocsize));
	softmax(output_.rightCols(conf_.redvocsize));

	return output_;
}

template<class Configuration>
const typename Configuration::Coeff
		LMEmbed<Configuration>::bprop(const Matrix &targets, const Coeff &W) {
	Coeff grads = Coeff::zeros(conf_);

	Matrix error = output_ - input_;
	grads.hidout().noalias() = hidden_.adjoint() * error;
	grads.outbias().noalias() = error.colwise().sum();

	Matrix hiderr = conf_.activation.df(hidin_, hidden_).cwiseProduct(error * W.hidout());
	grads.inphid().noalias() = input_.adjoint() * hiderr;
	grads.hidbias().noalias() = hiderr.colwise().sum();

	return grads;
}

template<class Derived>
void softmax(const Eigen::MatrixBase<Derived> &mc) {
	// const_cast hack endorsed by Eigen manual
	Eigen::MatrixBase<Derived> &m = const_cast<Eigen::MatrixBase<Derived> &>(mc);
	m.colwise() -= m.rowwise().maxCoeff();
	m = m.array().exp();
	m.array().colwise() /= m.rowwise().sum().array();
}

struct TrainingParameters {
	uint nsteps;
	uint nbatch;
	uint batchsize;
	Float alpha;
	Float momentum;

	TrainingParameters() {
		nsteps = 10;
		nbatch = 1000;
		batchsize = 32;
		alpha = .001;
		momentum = .9;
	}
};

template<class Configuration>
typename Configuration::Coeff nnopt(Configuration conf, TrigramSampler batch, Batch val,
		typename Configuration::Coeff W, TrainingParameters p) {
	typedef typename Configuration::Coeff Coeff;

	Coeff gain = Coeff::ones(conf);
	Coeff weight_change = Coeff::zeros(conf);
	Coeff prev_change = Coeff::zeros(conf);
	Coeff rms = Coeff::ones(conf);

	uint dotsteps = p.nbatch / 80;

	typename Configuration::Network net(conf);
	for(uint i = 0; i < p.nsteps; i++) {
		Float err = 0;

		for(uint j = 0; j < p.nbatch; j++) {
			if(j % dotsteps == 0)
				std::cerr << '.';

			Batch b = batch(p.batchsize);
			Array trainout = net.fprop(b.input, W);
			Coeff grads = net.bprop(b.targets, W);

			err -= (b.targets.array() * trainout.cwiseMax(TINY).log()).sum() /
				(p.nbatch * p.batchsize);

			rms.array() = Float(.9) * rms.array() + Float(.1) * grads.array().square();
			grads.array() /= (rms.array() + TINY).sqrt();

			weight_change.array() = p.momentum * weight_change.array() -
				p.alpha * gain.array() * grads.array();
			W.array() += weight_change.array();

			typedef Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> BoolArray;

			BoolArray changed_sign =
				((weight_change.array() < ZERO) && (prev_change.array() > ZERO) ||
				(weight_change.array() > ZERO) && (prev_change.array() < ZERO));
			BoolArray same_sign =
				((weight_change.array() < ZERO) && (prev_change.array() < ZERO) ||
				(weight_change.array() > ZERO) && (prev_change.array() > ZERO));

			gain.array() *= Float(1) - changed_sign.cast<Float>() * Float(.05);
			gain.array() += same_sign.cast<Float>() * Float(.05);
		}

		Array valout = net.fprop(val.input, W);
		Float valerr = -(val.targets.array() * valout.cwiseMax(TINY).log()).sum() / val.nitems;

		std::cerr << "\nTraining error: " << err << ", validation error: " << valerr << std::endl;
	}

	return W;
}

int main() {
	std::cerr << "Loading data." << std::endl;
	uint vocsize;
	TrigramCountMap tgmap = load_trigrams(vocsize);
	TrigramSampler sampler(vocsize, tgmap);
	tgmap.clear();

	const int VALSIZE = 10000;
	const int TESTSIZE = 10000;
	
	typedef LMEmbedConfiguration<SigmoidActivation> Config;
	Config conf(SigmoidActivation(), vocsize, 100, PREDICTION_VOCSIZE + 1);

	TrainingParameters params;

	Batch val = sampler.extract(VALSIZE);
	Batch test = sampler.extract(TESTSIZE);
	Config::Coeff W = nnopt(conf, sampler, val, Config::Coeff::init_weights(conf), params);

	Config::Network net(conf);
	Array testout = net.fprop(test.input, W);
	Float testerr = -(test.targets.array() * testout.cwiseMax(TINY).log()).sum() / test.nitems;
	std::cerr << "Test error: " << testerr << std::endl;

	for(uint i = 0; i < W.array().rows(); i++)
		std::cout << W.array()(i);

	return 0;
}
