%% BRYCE CHUNG 20200820


%% load data

n_files = -1;

files = dir('*.rhd');
if n_files < 0; n_files = length(files); end

t_data = [];
wav_data = [];

datafile_len = 0;
for i=1:n_files
    disp(files(i).name);    
    read_Intan_RHD2000_nongui(files(i).name);    
    disp('done loading...');

    if i==1
        datafile_len = length(t_board_adc);
        t_data = zeros(datafile_len*n_files,1);
        wav_data = zeros(datafile_len*n_files,1);
    end

    file_len = length(t_board_adc);
    
    start_ix = datafile_len*(i-1)+1;
    end_ix = start_ix+file_len-1;
    
    t_data(start_ix:end_ix) = t_board_adc;
    wav_data(start_ix:end_ix) = board_adc_data;
    
    if file_len < datafile_len
        del_start_ix = end_ix+1;
        del_end_ix = del_start_ix + (datafile_len - file_len)-1;
        
        wav_data(del_start_ix:del_end_ix) = [];
        t_data(del_start_ix:del_end_ix) = [];
    end
    
    clearvars -except files t_data wav_data datafile_len
    
    disp('processing complete!');
end

file_str = split(files(1).name, '_');
save([file_str{1} '_pressure_raw.mat'], 'wav_data', 't_data', '-v7.3');

clear all

disp('LOADED ALL DATA AND SAVED FILE!');

% ===== ===== ===== ===== =====
% process data to define behavioral cycles
% ===== ===== ===== ===== =====

%% CALIBRATE DATA
% --> THIS NEEDS TO BE REWRITTEN FOR GENERAL APPLICATION ACROSS
% DATASETS/EXPERIMENTS

% clear all;
dataset_name = 'bl21lb21'; 
dataset_smoothing = 7500 ;

disp(['Loading data for ' dataset_name]);
load([dataset_name '_pressure_raw.mat']);

switch dataset_name
    case 'bk12bk14'
        % bk12bk14 calibration
        inh20_kpa_conv = 0.149174;
        slope = 4.0760;
        first_file_mean = 1.5157;
        pressure_offset = 1.5157;
    case 'bl21lb21'
        % bl21lb21 calibration
        inh20_kpa_conv = 0.149174;
        slope = 15.4467;
        first_file_mean = 1.5336;
        pressure_offset = 1.5336;
    case 'gr43or93'
        % gr43or93 calibration
        inh20_kpa_conv = 0.149174;
        slope = 14.8046;
        first_file_mean = 1.5309;
        pressure_offset = 1.5320;
    case 'lb61rd26'
        % bl21lb21 calibration
        inh20_kpa_conv = 0.149174;
        slope = 14.3881;
        first_file_mean = 1.5298;
        pressure_offset = 1.5306;
    case 'bk17bl16'
        % bl21lb21 calibration
        inh20_kpa_conv = 0.149174;
        slope = 14.3881;
        first_file_mean = 1.5298;
        pressure_offset = 1.5306;
    case 'pu16sp16'
        inh20_kpa_conv = 0.149174;
        slope = 14.7277;
        first_file_mean = 1.5276;
        pressure_offset = 1.5291;
    case 'lb143pk8'
        inh20_kpa_conv = 0.149174;
        slope = 14.6774;
        first_file_mean = 1.5300;
        pressure_offset = 1.5305;
    case 'gr20rd33'
        inh20_kpa_conv = 0.149174;
        slope = 15.2008;
        first_file_mean = 1.5286;
        pressure_offset = 1.5298;
end


disp('calibrating raw behavior data...');
if size(wav_data,1) == 1
    cal_data = (inh20_kpa_conv*slope*((wav_data/mean(wav_data))*first_file_mean - pressure_offset))';
else
    cal_data = inh20_kpa_conv*slope*((wav_data/mean(wav_data))*first_file_mean - pressure_offset);
end

disp('applying lowpass filter...');
filt_data_tmp = bandpass_filtfilt(cal_data, 30000, 1, 5, 'hanningfir');

disp('smoothing data...');
filt_data = smoothdata(filt_data_tmp, 'gaussian' , dataset_smoothing); %try 300 first, then 3000, then 30000

disp('calibration and filtering complete!'); 

disp('plotting phase portrait');
figure();
plot(filt_data(1:end-1), diff(filt_data), 'b-');
%% THRESHOLD DATA
disp('processing behavior by threshold...');
cycle_th = 0;

th_xings = diff(sign(filt_data-cycle_th+1e-6)); % offset zero values
cycle_starts = find(th_xings == 2);
cycle_ends = find(th_xings == -2);

disp('filtering thresholds...');
if cycle_starts(1) > cycle_ends(1); cycle_ends(1) = []; end
if cycle_starts(end) > cycle_ends(end); cycle_starts(end) = []; end
assert(length(cycle_starts) == length(cycle_ends), 'ERROR: dimension mismatch cycle start and end');
cycle_times = [cycle_starts' cycle_ends'];


% % Preset threshold
cycle_int_th = 0.43;
cycle_dur_th = 0.15;

ints = diff(t_data(cycle_starts));
durs = diff([t_data(cycle_starts)'; t_data(cycle_ends)']);

% % User-defined threshold
% disp('Select the cycle interval threshold to filter data...');
% [cycle_int_th, ~] = ginput(1);

figure();
histogram(durs);
set(gca, 'yscale', 'log');
hold on;
ylims = get(gca, 'ylim');
plot([cycle_dur_th cycle_dur_th], ylims, 'r--');
title([dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
% savefig([dataset_name '_detection_histogram_' num2str(dataset_smoothing) '-threshold.fig']);

bad_cycles = find(durs < cycle_dur_th);
good_cycle_starts = cycle_starts;
good_cycle_starts(bad_cycles) = [];
good_cycle_ends = cycle_ends;
good_cycle_ends(bad_cycles) = [];


disp('plotting threshold validation...');
figure();
plot(t_data, filt_data, 'k-');
hold on;
xlims = get(gca, 'xlim');
plot(xlims, [cycle_th cycle_th], 'r--');

marker_offset = max(filt_data)*1.2;
marker_spacing = marker_offset * 0.25;

plot([t_data(cycle_starts)'; t_data(cycle_ends)'], marker_offset*[ones(length(cycle_starts),1) ones(length(cycle_ends),1)]', 'b-', 'LineWidth', 2);
plot([t_data(good_cycle_starts)'; t_data(good_cycle_ends)'], marker_offset*[ones(length(good_cycle_starts),1) ones(length(good_cycle_ends),1)]'+marker_spacing, 'm-', 'LineWidth', 2);
title([dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
% savefig([dataset_name '_detection_validation_' num2str(dataset_smoothing) '-threshold.fig']);


% figure();
% durs = diff([t_data(good_cycle_starts)'; t_data(good_cycle_ends)']);
% areas = arrayfun(@(x) sum(filt_data(good_cycle_starts(x):good_cycle_ends(x))), 1:length(good_cycle_starts));
% scatter(durs, areas, 5, 'b', 'filled');
% xlabel('Cycle Duration (s)');
% ylabel('Cycle Area (au)');


disp('getting waveforms...');
cycle_wavs = get_waveforms(filt_data_tmp, good_cycle_starts);

disp('resampling waveforms...');
cycle_wavs_mat = resample_waveforms(cycle_wavs, 100);

disp('plotting waveforms...');
figure();
plot(cycle_wavs_mat);
title([dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
% savefig([dataset_name '_waveforms_' num2str(dataset_smoothing) '-threshold.fig']);

[coeff, scores, latent] = pca(cycle_wavs_mat');
figure();
scatter(scores(:,1), scores(:,2), 'b.');
title([dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
% savefig([dataset_name '_pca_' num2str(dataset_smoothing) '-threshold.fig']);

disp('done!');

save([dataset_name '_smooth_pressure_data_' num2str(dataset_smoothing) '-threshold.mat'], 'cycle_wavs', 't_data', 'cycle_wavs_mat','cycle_times', 'filt_data_tmp', '-V7.3');

%% HILBERT TRANSFORM DATA
h_cycle_th = 0.2;

h_transform = hilbert(filt_data);
phase_data = atan2(imag(h_transform), real(h_transform));

h_cycle_starts = find(diff(sign(phase_data+1e-6)) == -2);
h_cycle_ends = find(diff(sign(phase_data+1e-6)) == 2);

if h_cycle_ends(1) < h_cycle_starts(1)
    h_cycle_starts = h_cycle_starts(1:end-1);
    h_cycle_ends = h_cycle_ends(2:end);
end


figure();
plot(cal_data, 'k-', 'LineWidth', 2);
hold on;
plot(imag(h_transform)/max(imag(h_transform))*max(filt_data), 'r-');
plot(real(h_transform)/max(real(h_transform))*max(filt_data), 'g-');
plot(phase_data/max(phase_data)*max(filt_data), 'm-');
plot([0 length(filt_data)], [0 0], 'k--');
scatter(h_cycle_starts, filt_data(h_cycle_starts), 25, 'm', 'filled');
scatter(h_cycle_starts, cal_data(h_cycle_starts), 50, 'b', 'filled');
title([dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
% savefig([dataset_name '_transform_validation_' num2str(dataset_smoothing) '-hilbert.fig']);

h_durs = diff([t_data(h_cycle_starts); t_data(h_cycle_ends)]);

figure()
histogram(h_durs);
hold on;
ylims = get(gca, 'ylim');
plot([h_cycle_th h_cycle_th], ylims, 'r--');
title([dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
% savefig([dataset_name '_detection_histogram_' num2str(dataset_smoothing) '-hilbert.fig']);

% good_h_cycles_ixs = find(h_durs > h_cycle_th);
% h_good_cycle_starts = h_cycle_starts(good_h_cycles_ixs);
% h_good_cycle_ends = h_cycle_ends(good_h_cycles_ixs);
% cycle_times = [h_good_cycle_starts h_good_cycle_ends];

h_good_cycle_starts = h_cycle_starts;
h_good_cycle_ends = h_cycle_ends;
cycle_times = [h_good_cycle_starts h_good_cycle_ends];


% figure()
% plot(t_data, filt_data, 'k-');
% hold on;
% xlims = get(gca, 'xlim');
% plot(xlims, [cycle_th cycle_th], 'r--');
% 
% marker_offset = max(filt_data)*1.2;
% marker_spacing = marker_offset * 0.25;
% 
% plot([t_data(h_cycle_starts)'; t_data(h_cycle_ends)'], marker_offset*[ones(length(h_cycle_starts),1) ones(length(h_cycle_ends),1)]', 'b-', 'LineWidth', 2);
% plot([t_data(h_good_cycle_starts)'; t_data(h_good_cycle_ends)'], marker_offset*[ones(length(h_good_cycle_starts),1) ones(length(h_good_cycle_ends),1)]'+marker_spacing, 'm-', 'LineWidth', 2);
% title([dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
% savefig([dataset_name '_detection_validation_' num2str(dataset_smoothing) '-hilbert.fig']);


disp('getting waveforms...');
cycle_wavs = get_waveforms(cal_data, h_good_cycle_starts);

disp('resampling waveforms...');
cycle_wavs_mat = resample_waveforms(cycle_wavs, 100);

disp('plotting waveforms...');
figure();
plot(cycle_wavs_mat);
title([dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
% savefig([dataset_name '_waveforms_' num2str(dataset_smoothing) '-hilbert.fig']);

[coeff, scores, ~,~,explained] = pca(cycle_wavs_mat');
figure();
scatter(scores(:,1), scores(:,2), 'b.');
title([dataset_name ' smoothing = ' num2str(dataset_smoothing)]);
% savefig([dataset_name '_pca_' num2str(dataset_smoothing) '-hilbert.fig']);

disp('done!');

save([dataset_name '_smooth_pressure_data_' num2str(dataset_smoothing) '-hilbert.mat'], 'cycle_wavs', 't_data', 'cycle_wavs_mat','cycle_times', 'filt_data_tmp', '-V7.3');
%%
% define clusters

global polypts % needs to be accessible within the callback function
polypts = [];

figure();
ax = subplot(1,1,1);
plot(scores(:,1), scores(:,2), 'k.', 'HitTest', 'off'); % plot data points
hold on;
set(ax, 'ButtonDownFcn', @user_cluster); % assign callback function
x = input('Set cluster: [Enter] | Cancel: [x]', 's'); % set up user keyboard input to indicate polygon is complete

if strcmp(x, 'x')
    polypts = []; % clear polygon if cancel
else
    plot([polypts(1,1) polypts(end,1)], [polypts(1,2) polypts(end,2)], 'r-'); % close polygon shape
end

[in, on] = inpolygon(scores(:,1), scores(:,2), polypts(:,1), polypts(:,2));
grp_pts = (in | on); % you can now recover a boolean of which points are in the selected cluster

scatter(scores(grp_pts,1), scores(grp_pts,2), 5, 'r', 'filled');


%% compare data

% Threshold method
% good_cycle_starts
% good_cycle_ends
%
% Hilbert method
% h_good_cycle_starts
% h_good_cycle_ends

marker_offset = max(filt_data)*1.2;
marker_spacing = marker_offset * 0.25;


figure();
plot(t_data, filt_data, 'k-');
hold on;


plot([t_data(cycle_starts)'; t_data(cycle_ends)'], marker_offset*[ones(length(cycle_starts),1) ones(length(cycle_ends),1)]', 'b-', 'LineWidth', 2);
plot([t_data(good_cycle_starts)'; t_data(good_cycle_ends)'], marker_offset*[ones(length(good_cycle_starts),1) ones(length(good_cycle_ends),1)]'+marker_spacing, 'r-', 'LineWidth', 2);

plot([t_data(h_cycle_starts)'; t_data(h_cycle_ends)'], marker_offset*[ones(length(h_cycle_starts),1) ones(length(h_cycle_ends),1)]'+3*marker_spacing, 'c-', 'LineWidth', 2);
plot([t_data(h_good_cycle_starts)'; t_data(h_good_cycle_ends)'], marker_offset*[ones(length(h_good_cycle_starts),1) ones(length(h_good_cycle_ends),1)]'+4*marker_spacing, 'm-', 'LineWidth', 2);

ylims = get(gca, 'ylim');
set(gca, 'ylim', [ylims(1) marker_offset + 5*marker_spacing]);

%%
% plot corresponding cycle start times
nearest_ixs = zeros(1,length(good_cycle_starts));
nearest_diff = zeros(1,length(good_cycle_starts));

for i=1:length(nearest_ixs)
%     [~, ix] = min(abs(t_data(h_good_cycle_starts)-t_data(good_cycle_starts(i))));
    [~, ix] = min(abs(x(:,1)-t_data(good_cycle_starts(i))));
    nearest_ixs(i) = ix;
%     nearest_diff(i) = t_data(good_cycle_starts(i)) - t_data(h_good_cycle_starts(ix));
    nearest_diff(i) = t_data(good_cycle_starts(i)) - x(ix,1);
end

% figure();
% xy = max([t_data(good_cycle_starts(end)) t_data(h_good_cycle_starts(end))]);
% plot([0 xy], [0 xy], 'k--');
% hold on;
% scatter(t_data(good_cycle_starts), t_data(h_good_cycle_starts(nearest_ixs)), 10, 'b', 'filled');
% xlabel('Threshold Cycle Start (s)');
% % ylabel('Hilbert Cycle Start (s)');


figure();
scatter(1:length(nearest_diff), nearest_diff, 10, 'b', 'filled');
hold on;
xlims = get(gca, 'xlim');
% plot(xlims, [0 0], 'k--');
xlabel('Threshold Cycle (ID)');
ylabel('Difference of Cycle Start (s)');

%%
% plot corresponding cycle durations
durs = diff([t_data(good_cycle_starts)'; t_data(good_cycle_ends)']);
nearest_durs = diff([t_data(h_good_cycle_starts(nearest_ixs))'; t_data(h_good_cycle_ends(nearest_ixs))']);

figure();
scatter(durs, nearest_durs, 10, 'b', 'filled');
hold on;
xlims = get(gca, 'xlim');
ylims = get(gca, 'ylim');
plot([0 max([xlims(2) ylims(2)])], [0 max([xlims(2) ylims(2)])], 'k--');
xlabel('Threshold Cycle Duration (s)');
ylabel('Hilbert Cycle Duration (s)');

figure();
scatter(1:length(durs), durs-nearest_durs, 10, 'b', 'filled');
xlabel('Threshold Cycle (ID)');
ylabel('Difference of Cycle Duration (s)');


%%
function cycle_wavs = get_waveforms(wav_data, starts, varargin)

    p = inputParser;
    addRequired(p, 'wav_data');
    addRequired(p, 'starts');
    addOptional(p, 'ends', []);
    parse(p, wav_data, starts, varargin{:});

    disp('processing data...');
    if length(p.Results.ends) > 0
        assert(length(starts) == length(ends), 'Dimension mismatch: starts & ends');
        disp('getting waveforms from durations...');
        
    else
        disp('getting waveforms from intervals...');
        cycle_wavs = arrayfun(@(i) wav_data(starts(i):starts(i+1)), 1:(length(starts)-1), 'UniformOutput', false);
    end
end

function cycle_wavs_resamp = resample_waveforms(cycle_wavs, sample_n)
    tmp_out = arrayfun(@(i) resample(cycle_wavs{i}, sample_n, length(cycle_wavs{i}),0), 1:length(cycle_wavs), 'UniformOutput', false);
    cycle_wavs_resamp = cell2mat(tmp_out);
end

function user_cluster(s,e)
    global polypts

    plot(e.IntersectionPoint(1), e.IntersectionPoint(2), 'r+', 'MarkerSize', 10);
    
    if length(polypts) == 0
        polypts = e.IntersectionPoint(1:2);
    else
        polypts(end+1,:) = e.IntersectionPoint(1:2);
        plot([polypts(end-1,1) polypts(end,1)], [polypts(end-1,2) polypts(end,2)], 'r-');
    end

end