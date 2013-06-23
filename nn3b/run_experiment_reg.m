function run_experiment(l2regulariser, l1regulariser)
	srcembed = 10;
	antembed = 10;
	hidden = 10;
	net = setup_net3(srcembed, antembed, hidden, {{'ce','c'''},{'elle'},{'elles'},{'il'},{'ils'}});
	net.regulariser = l2regulariser;
	net.l1regulariser = l1regulariser;
	[data,net] = load_net3_data(net, 'ted+ncv6.antf.nn3', 'ted-test.antf.nn3');
	[targets,valtgt,testtgt] = targets_net3(net, data);
	[weights,trainerr,valerr,bestve] = nnopt_net3(net, data, targets, valtgt);
	testout = fprop_net3(net, data.test, weights);
	PRF = pr(testout, testtgt)

	outfile = sprintf('tedncantf-reg-%f-%f.mat', l1regulariser, l2regulariser);

	save(outfile);
end
