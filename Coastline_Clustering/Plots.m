clear; close all; clc

% colors
rgb_o= [0 0 0;
    0.6953    0.1328    0.1328;
    1.0000    0.8398         0;
    0    0.3906         0;
    0         0    0.8008;
    0.9297    0.5078    0.9297;
    1.0000    0.5469         0];

% data
xx = load ('Clustering_MatchLevel.mat');

fields = fieldnames(xx);

for i = 1: length(fields)
    
    xi = xx.(fields{i});
    
    % all coordinates
    coordinates = [];
    for ii = 1: size(xi,1)
        coordinates = cat(1,coordinates,[cell2mat(xi.cluster_Model(ii));...
            cell2mat(xi.cluster_Obs(ii))]);
    end
    
    hh = figure;
    hh.Position = 1.0e+03* [0.0010    0.0410    1.5360    0.7488];
    
    for ii = 1:2
        subplot(1,2,ii); hold all;
        box on
        
        m_proj('miller','long',[min(coordinates(:,1))-1 max(coordinates(:,1))+1],...
            'lat',[min(coordinates(:,2))-1 max(coordinates(:,2))+1]);
        
        m_coast('patch',[.8 .8 .8],'edgecolor','none');
        m_grid('tickdir','in','linest','none') % ,'yticklabels',[],'xticklabels',[]);
    end
    
    for ii = 1: size(xi,1)
        
        cluster   = cell2mat(xi.cluster_Obs(ii));
        centroidi = cell2mat(xi.centroid_Obs(ii));
        
        % plot observations
        subplot(1,2,1)
        m_line(cluster(:,1),cluster(:,2),...
            'Color',rgb_o(ii,:),...
            'Marker','.','Linest','none','markersize',12)
        
        m_line(centroidi(:,1),centroidi(:,2),...
            'Color',rgb_o(ii,:),...
            'Marker','o','Linest','none','LineWidth',2,'markersize',7)
        
        title('Observations')
        
        cluster   = cell2mat(xi.cluster_Model(ii));
        centroidi = cell2mat(xi.centroid_Model(ii));
        
        % plot simulations
        subplot(1,2,2)
        m_line(cluster(:,1),cluster(:,2),...
            'Color',rgb_o(ii,:),...
            'Marker','.','Linest','none','markersize',12)
        
        m_line(centroidi(:,1),centroidi(:,2),...
            'Color',rgb_o(ii,:),...
            'Marker','d','Linest','none','LineWidth',1.5,'MarkerSize',7)
        
        title('Simulations')
        
    end
 
end









