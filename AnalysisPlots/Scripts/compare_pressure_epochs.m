%% Identify pressure cycle epochs
% addpath('ContinuousMIEstimation');
dataset_name = 'gr43or93' ;
dataset_smoothing = 7500;

epoch_window = 250;


switch dataset_name 
    case 'bk12bk14'
        load(['bk12bk14_smooth_pressure_data_' num2str(dataset_smoothing) '-hilbert.mat']);
        load('bk12bk14_20210805_dataObjs.mat', 'd');
        resetting = false;
    case 'bl21lbl21'
        load(['bl21lb21_smooth_pressure_data_' num2str(dataset_smoothing) '-hilbert.mat']);
        load('20200427_bl21lb21_07152020data.mat', 'd');
        resetting = true;
    case 'bk17bl16'
        load(['bk17bl16_smooth_pressure_data_' num2str(dataset_smoothing) '-hilbert.mat']);
        resetting = true;
    case 'gr20rd33'
        load(['gr20rd33_smooth_pressure_data_' num2str(dataset_smoothing) '-hilbert.mat']);
        resetting = true;
    case 'lb143pk8'
        load(['lb143pk8_smooth_pressure_data_' num2str(dataset_smoothing) '-hilbert.mat']);
        resetting = true;
    case 'gr43or93'
        load(['gr43or93_smooth_pressure_data_' num2str(dataset_smoothing) '-hilbert.mat']);
        resetting = false;
    case 'lb61rd26'
        load(['lb61rd26_smooth_pressure_data_' num2str(dataset_smoothing) '-hilbert.mat']);
        resetting = false;
end




pts = 100;
std_th = 5;

cycle_dur = cellfun(@(x) length(x), cycle_wavs);
mm = movmean(cycle_dur, pts);
ms = movstd(cycle_dur, pts);

if resetting
    cycle_ixs = find(diff(sign(cycle_dur - mm - std_th*ms)) == 2)+1;
else
    cycle_ixs = 1:250:(length(cycle_wavs));
end


figure();
plot(cycle_dur);
hold on;
plot(cycle_dur, 'b.');
plot(mm);
plot(mm + std_th*ms);
scatter(cycle_ixs, cycle_dur(cycle_ixs), 25, 'r', 'filled');
title(['Cycle Duration - ' dataset_name ' smoothing = ' num2str(dataset_smoothing)]);

xlabel('Cycle ID');
ylabel('Duration');

%% Use peak detection of cycle maximum
%  This method identifies the lowest maximum but does not necessarily
%  identify the first cycle of the reset
%
% pk_th = 0.945;
% 
% cycle_max = cellfun(@(x) max(x), cycle_wavs);
% pks_bool = (1 - cycle_max) < pk_th;
% pks_raw = 1-cycle_max;
% pks_raw(find(pks_bool==1)) = 0;
% [~, pks_ix] = findpeaks(smoothdata(pks_raw, 'gaussian', 5));
% 
% 
% figure();
% plot(1-cycle_max);
% hold on;
% plot(pks_raw);
% scatter(pks_ix, pks_raw(pks_ix), 25, 'r', 'filled');

%% Plot cycles within each epoch

epoch=1;
centroid_omit = 10;

[~, scores, ~] = pca(cycle_wavs_mat');

for epoch=1:(length(cycle_ixs)-1)
    figure();
    scatter(scores(:,1), scores(:,2), 5, 'k', 'filled', 'MarkerFaceAlpha', 0.1);
    xlabel('PC1');
    ylabel('PC2');
    title(['Epoch ' num2str(epoch) ' - ' dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
    hold on;

    plot(scores(cycle_ixs(epoch):(cycle_ixs(epoch+1)-1),1), scores(cycle_ixs(epoch):(cycle_ixs(epoch+1)-1),2), 'b.-', 'MarkerSize', 15);
    plot(scores(cycle_ixs(epoch),1), scores(cycle_ixs(epoch),2), 'bx', 'MarkerSize', 15);
    plot(mean(scores((cycle_ixs(epoch)+centroid_omit):(cycle_ixs(epoch+1)-1),1)), mean(scores(cycle_ixs(epoch):(cycle_ixs(epoch+1)-1),2)), 'rx', 'MarkerSize', 10);

    figure();
    plot(cycle_wavs_mat(:,cycle_ixs(epoch):cycle_ixs(epoch+1)-1), 'k-');
    hold on;

    wavs_mean = mean(cycle_wavs_mat(:,cycle_ixs(epoch):cycle_ixs(epoch+1)-1)');
    wavs_std = std(cycle_wavs_mat(:,cycle_ixs(epoch):cycle_ixs(epoch+1)-1)');
    plot(wavs_mean, 'r-', 'LineWidth', 3);
    plot(wavs_mean+wavs_std, 'c-', 'LineWidth', 2);
    plot(wavs_mean-wavs_std, 'c-', 'LineWidth', 2);
    ylim([-0.15 0.2]);
    xlabel('Resample ID');
    ylabel('Pressure');
    title(['Epoch ' num2str(epoch) ' - ' dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
end


%% build neural rasters

epoch = 1;
do_wavs = false;
do_hist = true;

if do_wavs
    pts_per_wav = 100;
    unit_raster_spacing = 1;
    wav_raster_height = 0.5;
end

%if Error arises, make sure that the ContinuousMIEstimation-master folder
%is on the search path
units = fields(d.data);
% units = units(1:3);
% units = units(4:6);
% units = units(1:2);
% units = units(4); % or any index from 1 to 6

neural_counts = cell(length(cycle_ixs)-1, length(units), 1);
t_elapsed = [ ];
for i=1:(length(cycle_ixs)-1)
    epoch = i;
    disp([newline 'plotting epoch ' num2str(epoch) ' of ' num2str(length(cycle_ixs))]);
    tic;
    if do_wavs
        wav_raster_spacing = wav_raster_height/(cycle_ixs(epoch+1) - cycle_ixs(epoch));
        figure();
    end
    for u=1:length(units)
        for e=1:(cycle_ixs(epoch+1)-cycle_ixs(epoch))
            cycle_ix = cycle_ixs(epoch)+e-1;
            %for each neuron iterating through each cycle
%             cycle_lims = t_data([cycle_times(cycle_ix) cycle_times(cycle_ix+1)],1);
            cycle_lims = t_data([cycle_times(cycle_ix) cycle_times(cycle_ix+1)]);

            raster_ixs = (cycle_lims(1) <= d.data.(units{u}).data) & (cycle_lims(2) >= d.data.(units{u}).data);
            neural_counts{i,u}(end+1) = sum(raster_ixs);
            
                if do_wavs
                    t_scaled = (d.data.(units{u}).data(raster_ixs)-cycle_lims(1))/(cycle_lims(2)-cycle_lims(1));
                    scatter(t_scaled*pts_per_wav, ones(1,sum(raster_ixs)).*((u)*unit_raster_spacing+(e-1)*wav_raster_spacing), 3, 'k', 'filled', 'MarkerFaceAlpha', 0.1);
                    hold on;
                end
        end
        if do_wavs
            xlim([0 pts_per_wav]);
            ylim([0 length(units)+1]);
        end
    end
    if do_wavs
        title(['Epoch ' num2str(epoch) ' - ' dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
        drawnow();
        t_elapsed(end+1) = toc;
        toc;
        t= seconds(mean(t_elapsed)*(length(cycle_ixs)-i));
        disp(['estimated time remaining: ' num2str(floor((minutes(t)))) ':' num2str(seconds((t - seconds(floor(minutes(t))*60))))]);

        %disp(['estimated time remaining: ' num2str((mean(t_elapsed)*(length(cycle_ixs) - i))) 's']);
    end
end

hist_data = zeros(length(units), length(cycle_ixs));
hist_std = zeros(length(units), length(cycle_ixs));
bar_pos = 1;
x_labels = {'0'};
if do_hist
%     epoch_types = {[1 19 20 21 28 29 30 31], [2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 23 24 25 26 27 32 33]};
    epoch_types = {[1:40]};
    
    for t=1:length(epoch_types)
        epochs = epoch_types{t};
        for e=1:length(epochs)
            for u=1:length(units)
                hist_data(u,bar_pos) = mean(neural_counts{epochs(e),u});
                hist_std(u,bar_pos) = std(neural_counts{epochs(e),u});
            end
            x_labels{1+bar_pos} = num2str(epochs(e));
            bar_pos = bar_pos + 1;
        end
        hist_data(:,bar_pos) = 0;
        x_labels{1+bar_pos} = '0';
        bar_pos = bar_pos + 1;
    end
    figure();
    bplot = bar(hist_data');
    hold on;
    
    if length(units) == 1
        group_spacing = 0.7; % 2 units
        offset = 0; % only one unit
        bar_space = 0.15; % 2 units
    elseif length(units) == 2
        group_spacing = 0.7; % 2 units
        offset = (1-group_spacing)/size(hist_std,1);
        bar_space = 0.15; % 2 units
    elseif length(units) == 3
        group_spacing = 0.55; % 3 units
        offset = (1-group_spacing)/size(hist_std,1);
        bar_space = 0.085; % 3 units
    else
        group_spacing = 1/3.; % all 6 units
        offset = (1-group_spacing)/size(hist_std,1);
        bar_space = 0.02; % all 6 units
    end    
    
    for t=1:length(epoch_types);
        epochs = epoch_types{t};
        for e=1:length(epochs)
            for u=1:length(units)
                first_pos = epochs(e) - length(units)*offset/2.;
                plot([first_pos first_pos]+(offset+bar_space)*(u-1), [0 hist_std(u,epochs(e))] + hist_data(u,epochs(e)), 'k-');
            end
        end
    end
    
    xlim([-1 length(cycle_ixs)]);
    xticks(0:length(cycle_ixs));
    xticklabels(x_labels);
    
    title([ dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
    ylabel('Spike Counts by Unit #');
    xlabel('Epoch # by Cluster');
end
%% Plot neural spike count based on pressure waveform PC1,PC2 plot

% collapse neural counts within unit to rescale colormap
count_span = zeros(2,size(neural_counts,2));
count_mat = {};

for i=1:size(count_span,2)
    count_span(1,i) = min(cellfun(@(x) min(x), neural_counts(:,i)));
    count_span(2,i) = max(cellfun(@(x) max(x), neural_counts(:,i)));
    
    count_mat{i} = neural_counts{1,i};
    for j=2:size(neural_counts,1)
        count_mat{i}(end+1:end+length(neural_counts{j,i})) = neural_counts{j,i};
    end
end

tmp = zeros(size(neural_counts,2), size(count_mat{1},2));
for i=1:size(tmp,1)
%     tmp(i,:) = count_mat{i}/diff(count_span(:,i));
    tmp(i,:) = count_mat{i};
end
count_mat = tmp;

count_th = 50;
plot_th = 1;
for i=1:size(count_mat,1)
    
    plot_ixs = count_mat(i,:) >= plot_th;
    
    % set up color map
    if i==1 || i==3
        cm = jet(count_th);
        cs = arrayfun(@(x) cm(min(x,count_th),:), count_mat(i,plot_ixs), 'UniformOutput', false);
        cb_min = max(min(count_mat(i,plot_ixs),plot_th));
        cb_max = min(max(count_mat(i,plot_ixs),count_th));
    else
        cm = jet(max(count_mat(i,plot_ixs))-min(count_mat(i,plot_ixs))+1); % need to check on consistency of range
        cs = arrayfun(@(x) cm(x,:), count_mat(i,plot_ixs), 'UniformOutput', false);
        cb_min = min(min(count_mat(i,plot_ixs)));
        cb_max = max(max(count_mat(i,plot_ixs)));
    end
    
    c = reshape(cell2mat(cs),3,length(cs));
    

    plot_ixs(end+1:end+(size(scores,1)-length(plot_ixs))) = false;
    % plot points
    figure();
    s = scatter(scores(plot_ixs,1), scores(plot_ixs,2), 10, c', 'filled'); % last cycle is omitted to find spike counts
    s.MarkerFaceAlpha = 0.2;
    title(['Unit ' num2str(i) ', count > 0 - ' dataset_name ' smoothing= ' num2str(dataset_smoothing)]);
    xlabel('PC1');
    ylabel('PC2');
    cb = colorbar;
    cb.Ticks = [0 1];
    cb.TickLabels = [cb_min cb_max];
    drawnow();
    
    figure();
    histogram(count_mat(i,:)-1);
    title(['Unit ' num2str(i) ' - ' dataset_name ' smoothing= ' num2str(dataset_smoothing)]);
    xlabel('Spike Count');
    ylabel('Frequency');
    if i==1 || i==3
        ys = get(gca, 'ylim');
        hold on;
        plot([count_th count_th], ys, 'r--');
    end
    xlim([-2 120]);
    drawnow();
    disp('done!');
end
