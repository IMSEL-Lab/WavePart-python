set(0, 'DefaultFigureVisible', 'off');

repo_dir = fileparts(fileparts(mfilename('fullpath')));
assets_dir = fullfile(repo_dir, 'reports', 'assets');
if ~exist(assets_dir, 'dir')
    mkdir(assets_dir);
end

addpath(repo_dir);
load(fullfile(repo_dir, 'data', 'wavespec2d_ex.mat'));

idx_list = [1 287];
brand = [
    255 255 255
    115   0  10
    70  106 159
    31  65  77
    101 120  11
    164 145  55
    92  92  92
    54  54  54
] ./ 255;

for ii = 1:length(idx_list)
    idx = idx_list(ii);
    E0 = S(:, :, idx);
    [AA, Ef] = partition(freq, dir, E0, 1);
    np = max(AA(:));
    cmap = brand(1:(np + 1), :);

    fig = figure('Position', [100 100 1000 700], 'Color', [1 1 1]);
    ax = axes(fig);
    imagesc(ax, freq, dir, AA');
    axis(ax, 'xy');
    hold(ax, 'on');
    contour(ax, freq, dir, Ef', 7, 'LineColor', [0 0 0], 'LineWidth', 0.75);
    hold(ax, 'off');
    colormap(ax, cmap);
    clim(ax, [0 np]);
    cb = colorbar(ax);
    cb.Ticks = 0:np;
    cb.Color = [0 0 0];
    title(ax, sprintf('MATLAB Partition Surface. Index %d', idx), 'Color', [0 0 0]);
    xlabel(ax, 'Frequency (Hz)', 'Color', [0 0 0]);
    ylabel(ax, 'Direction (deg)', 'Color', [0 0 0]);
    set(ax, 'LineWidth', 1.0, 'Box', 'on', 'Layer', 'top', 'FontName', 'Helvetica');
    exportgraphics(fig, fullfile(assets_dir, sprintf('matlab_case_%03d_surface.png', idx)), 'Resolution', 180);
    close(fig);

    fig = figure('Position', [100 100 1100 760], 'Color', [1 1 1]);
    ax = axes(fig);
    surf(ax, freq', dir', Ef', AA', 'EdgeColor', 'none', 'FaceColor', 'interp');
    view(ax, [-38 32]);
    colormap(ax, cmap);
    clim(ax, [0 np]);
    cb = colorbar(ax);
    cb.Ticks = 0:np;
    cb.Color = [0 0 0];
    title(ax, sprintf('MATLAB Partition 3D View. Index %d', idx), 'Color', [0 0 0]);
    xlabel(ax, 'Frequency (Hz)', 'Color', [0 0 0]);
    ylabel(ax, 'Direction (deg)', 'Color', [0 0 0]);
    zlabel(ax, 'Smoothed Energy', 'Color', [0 0 0]);
    grid(ax, 'on');
    set(ax, 'LineWidth', 1.0, 'Box', 'on', 'FontName', 'Helvetica', 'GridColor', [0.35 0.35 0.35]);
    exportgraphics(fig, fullfile(assets_dir, sprintf('matlab_case_%03d_3d.png', idx)), 'Resolution', 180);
    close(fig);

    fig = figure('Position', [100 100 900 900], 'Color', [1 1 1]);
    ax = polaraxes(fig);
    delete(ax);
    [~, c] = polarPcolor(freq', [dir; dir(1)]', [AA AA(:, 1)]');
    colormap(cmap);
    c.Ticks = 0:np;
    c.Color = [0 0 0];
    title(sprintf('MATLAB Partition Polar. Index %d', idx), 'Color', [0 0 0]);
    exportgraphics(fig, fullfile(assets_dir, sprintf('matlab_case_%03d_polar.png', idx)), 'Resolution', 180);
    close(fig);
end
