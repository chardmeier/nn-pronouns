function dump_model(outstem, net, W)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    id = fopen(strcat(outstem, '.nn6.srcvoc'), 'w');
    fprintf(id, '%s\n', net.srcvoc{:});
    fclose(id);

    id = fopen(strcat(outstem, '.nn6.tgtvoc'), 'w');
    fprintf(id, '%s\n', net.srcvoc{:});
    fclose(id);

    if isfield(net, 'tgtprons')
        tgtprons = net.tgtprons;
    else
        tgtprons = {{'ce','c'''},{'elle'},{'elles'},{'il'},{'ils'}};
    end
    
    id = fopen(strcat(outstem, '.nn6.classes'), 'w');
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
    
    id = fopen(strcat(outstem, '.nn6.model'), 'w');
    header = [hex2dec('2fed70b9') 6 1 ...
                net.srcngsize length(net.srcvoc) length(net.tgtvoc) ...
                net.srcembed net.antembed net.hidden net.output];
    fwrite(id, header, 'uint32');
    
    fwrite(id, W.srcembed, 'float32');
    fwrite(id, W.antembed, 'float32');
    fwrite(id, W.embhid, 'float32');
    fwrite(id, W.hidout, 'float32');
    fclose(id);  
end

