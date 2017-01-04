function dump_model(outstem, net, voc, W)
% Save a trained model to disk.

    id = fopen(strcat(outstem, '.nn6.srcvoc'), 'w', 'n', 'UTF-8');
    fprintf(id, '%s\n', net.srcvoc{:});
    fclose(id);

    id = fopen(strcat(outstem, '.nn6.tgtvoc'), 'w', 'n', 'UTF-8');
    fprintf(id, '%s\n', net.tgtvoc{:});
    fclose(id);

    if isfield(net, 'tgtprons')
        tgtprons = net.tgtprons;
    else
        tgtprons = {{'ce','c'''},{'elle'},{'elles'},{'il'},{'ils'}};
    end
    
    id = fopen(strcat(outstem, '.nn6.classes'), 'w', 'n', 'UTF-8');
    for i = 1:length(tgtprons)
        fprintf(id, '%s ', tgtprons{i}{1:end-1});
        fprintf(id, '%s\n', tgtprons{i}{end});
    end
    fclose(id);
    
    % Header fields:
    % - magic number
    % - nn6 identifier
    % - file format version
    % - source n-gram size
    % - source vocabulary size
    % - target vocabulary size
    % - source embedding size
    % - antecedent embedding size
    % - hidden layer size
    % - output layer size
    
    if isfield(net, 'sample_antfeatures') && net.sample_antfeatures
        %version = 2;   % NADA disabled
        %version = 3;    % NADA enabled
        version = 4;    % NADA enabled, softmax fixed
    else
        version = 1;
    end
    
    id = fopen(strcat(outstem, '.nn6.model'), 'w', 'n', 'UTF-8');
    header = [hex2dec('2fed70b9') 6 version ...
                net.srcngsize length(net.srcvoc) length(net.tgtvoc) ...
                net.srcembed net.antembed net.hidden net.output];
    fwrite(id, header, 'uint32');
    
    fwrite(id, W.srcembed, 'float32');
    fwrite(id, W.antembed, 'float32');
    fwrite(id, W.embhid, 'float32');
    fwrite(id, W.hidout, 'float32');
    fclose(id);
    
    coref_file = strcat(outstem, '.nn6.coref');
    coref = struct('features', find(voc.activelinkfeat), 'Ahid', net.Ahid, ...
        'linkAhid', W.linkAhid, 'AhidAres', W.AhidAres');
    save(coref_file, '-struct', 'coref', '-ascii', '-tabs');
end

