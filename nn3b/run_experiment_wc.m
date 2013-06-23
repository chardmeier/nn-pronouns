function run_experiment_wc(constraint)
	srcembed = 10;
	antembed = 10;
	hidden = 20;
	net = setup_net3(srcembed, antembed, hidden, {{'ce','c'''},{'elle'},{'elles'},{'il'},{'ils'}});
	net.weightconstraint = constraint;
	%net.coupled_hidout = {{[2,3],[4,5]}, {[2,4],[3,5]}};
	%[data,net] = load_net3_data(net, 'tedtrain+ncv6+epv6.nn3', 'ted-test.nn3');
	%[data,net] = load_net3_data(net, 'ted+ncv6+epv6.antf.nn3', 'ted-test.antf.nn3');
	%[data,net] = load_net3_data(net, 'ted-train.nn3', 'ted-test.nn3');
	%[data,net] = load_net3_data(net, 'ted-train.antf.nn3', 'ted-test.antf.nn3');
	[data,net] = load_net3_data(net, 'ted+ncv6.antf.nn3', 'ted-test.antf.nn3');
	[targets,valtgt,testtgt] = targets_net3(net, data);
	[weights,trainerr,valerr,bestve] = nnopt_net3(net, data, targets, valtgt);
	testout = fprop_net3(net, data.test, weights);
	PRF = pr(testout, testtgt)

	outfile = sprintf('n10-20tedncantf-wc-%f.mat', constraint);

	save(outfile);
end
