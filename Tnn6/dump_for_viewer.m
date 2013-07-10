function dump_for_viewer(outfile, test, testout, internal)
    id = fopen(outfile, 'w');
    fprintf(id, 'ce\telle\telles\til\tils\tOTHER\n');
    fprintf(id, 'SOLUTIONS\n');
    fprintf(id, '%g %g %g %g %g %g\n', test.targets');
    fprintf(id, 'PREDICTIONS\n');
    fprintf(id, '%g %g %g %g %g %g\n', testout');
    fprintf(id, 'ANTECEDENT SCORES\n');
    for i = 1:length(test.antidx)
        fprintf(id, '%g ', internal.Ares(test.antidx{i}));
        fprintf(id, '\n');
    end
    fclose(id);
end