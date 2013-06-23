function run_experiment(srcembed, antembed, hidden, regulariser)
	net = setup_net3(srcembed, antembed, hidden, {{'ce','c'''},{'elle'},{'elles'},{'il'},{'ils'}});
	net.regulariser = regulariser;
	[data,net] = load_net3_data(net, 'extract.nn2');
	[targets,valtgt,testtgt] = targets_net3(net, data);
	[weights,trainerr,valerr,bestve] = nnopt_net3(net, data, targets, valtgt);
	testout = fprop_net3(net, data.test, weights);
	PRF = pr(testout, testtgt)

	outfile = sprintf('s%d-a%d-h%d-r%f.mat', srcembed, antembed, hidden, regulariser);
	save(outfile);
end
