function vis_mnist_images(data, varargin)
% works for MatConvNet MNIST images
opts.pairwise = false;
opts.numImg = 25;
opts = vl_argparse(opts, varargin) ;


figure                                          % plot images
colormap(gray)                                  % set to grayscale

numImg = opts.numImg; 

if ~iscell(data)
    if ~opts.pairwise
        numImg = min(25, size(data,4));
        for i = 1:numImg                                % preview first 25 samples
            subplot(5,5,i)                              % plot them in 6 x 6 grid
            digit = uint8(data(:,:,:,i));               % row = 28 x 28 image
            imagesc(digit)                              % show the image
            set(gca,'xtick',[]);
            set(gca,'ytick',[]);
            % title(num2str(tr(i, 1)))                  % show the label
        end
    else
       % pairwise images
       numImg = min(20, size(data,4));
       for i = 1:numImg                                 % preview first 10 pairs
            subplot(2, 10, (i - floor(i/2)) + 10*(~mod(i,2)))                             % plot them in 6 x 6 grid
            digit = uint8(data(:,:,:,i));               % row = 28 x 28 image
            imagesc(digit)                              % show the image
            axis('square')
            % axis('square')
            set(gca,'xtick',[]);
            set(gca,'ytick',[]);
            % title(num2str(tr(i, 1)))                  % show the label
        end 
    end
else
    numImg = min(25, length(data));
    for i = 1:numImg                                % preview first 25 samples
        subplot(5,5,i)                              % plot them in 6 x 6 grid
        digit = (data{i});                          % row = 28 x 28 image
        imagesc(digit)    
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        colormap default
        colorbar
        % colormap gray
        % title(num2str(tr(i, 1)))                  % show the label
    end 
end