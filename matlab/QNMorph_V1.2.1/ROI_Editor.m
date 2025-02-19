classdef ROI_Editor < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure              matlab.ui.Figure
        DONEButton            matlab.ui.control.Button
        SaveFinalImageButton  matlab.ui.control.Button
        SaveMaskButton        matlab.ui.control.Button
        SaveROIButton         matlab.ui.control.Button
        FillInsideButton      matlab.ui.control.Button
        ClearoutsideButton    matlab.ui.control.Button
        ClearinsideButton     matlab.ui.control.Button
        DrawPolygonButton     matlab.ui.control.Button
        DrawRectangleButton   matlab.ui.control.Button
        DrawCircleButton      matlab.ui.control.Button
        DrawFreehandButton    matlab.ui.control.Button
        DrawEllipseButton     matlab.ui.control.Button
        UpdateROIButton       matlab.ui.control.Button
        LoadROIButton         matlab.ui.control.Button
        UIAxes                matlab.ui.control.UIAxes
        UIAxes2               matlab.ui.control.UIAxes
        im;
        imf;
        ROI;
        infilename;
    end

    properties (Access = private)
        currentmask;
        oldmask;
        mask;
    end
    
    methods (Access = public)
                 function app = ROI_Editor(IM,Infilename)    
                     app.infilename=Infilename;
            if size(IM,3) == 3
                app.im = im2double(rgb2gray(IM));
            else
                app.im = im2double(IM);
            end
            app.ROI=gobjects(0);
            app.mask=zeros(size(app.im));
            % invoke the UI window
            app.createComponents;            
            % load the image
             imshow(imadjust(app.im),'Parent',app.UIAxes)
             imshow(imadjust(app.im),'Parent',app.UIAxes2)
             linkaxes([app.UIAxes,app.UIAxes2],'xy');
             app.imf=app.im;
         end
        
        
        function plotinput(app)
            % plot input image          
            imshow(imadjust(app.im),'Parent',app.UIAxes)
        end
        
        function plotout(app) %%% plot out put image
            imshow(app.imf,'Parent',app.UIAxes2)
        end 
                      

    end

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: DrawEllipseButton
        function DrawEllipseButtonPushed(app, event)
            h=drawellipse(app.UIAxes);
            app.ROI(end+1)=h;
            app.mask=app.mask|createMask(h);
            app.imf=im2double(app.mask).*im2double(app.im);
            plotout(app);
        end

        % Button pushed function: DrawCircleButton
        function DrawCircleButtonPushed(app, event)
            h=drawcircle(app.UIAxes);
            app.ROI(end+1)=h;
            app.mask=app.mask|createMask(h);
            app.imf=im2double(app.mask).*im2double(app.im);
            plotout(app);
        end

        % Button pushed function: DrawFreehandButton
        function DrawFreehandButtonPushed(app, event)
            h=drawfreehand(app.UIAxes);
            app.ROI(end+1)=h;
            app.mask=app.mask|createMask(h);
            app.imf=im2double(app.mask).*im2double(app.im);
            plotout(app);
        end

        % Button pushed function: DrawRectangleButton
        function DrawRectangleButtonPushed(app, event)
            h=drawrectangle(app.UIAxes);
            app.ROI(end+1)=h;
            app.mask=app.mask|createMask(h);
            app.imf=im2double(app.mask).*im2double(app.im);
            plotout(app);
        end

        % Button pushed function: DrawPolygonButton
        function DrawPolygonButtonPushed(app, event)
            h=drawpolygon(app.UIAxes);
            app.ROI(end+1)=h;
            app.mask=app.mask|createMask(h);
            app.imf=im2double(app.mask).*im2double(app.im);
            plotout(app);   
        end

        % Button pushed function: ClearinsideButton
        function ClearinsideButtonPushed(app, event)
            app.imf=im2double(imcomplement(app.mask)).*im2double(app.im);
            plotout(app);  
        end

        % Button pushed function: ClearoutsideButton
        function ClearoutsideButtonPushed(app, event)
            app.imf=im2double(app.mask).*im2double(app.im);
            plotout(app); 
        end

        % Button pushed function: FillInsideButton
        function FillInsideButtonPushed(app, event)
           app.imf=imcomplement(app.mask).*im2double(app.im)+app.mask;
            plotout(app);
        end

        % Button pushed function: DONEButton
        function DONEButtonPushed(app, event)
            delete(app.UIFigure)
        end

        % Button pushed function: UpdateROIButton
        function UpdateROIButtonPushed(app, event)
            app.ROI = findobj(app.UIAxes, 'Type', 'images.roi');
            editedMask = false(size(app.im));
        for ind = 1:numel(app.ROI)
             % Accumulate the mask from each ROI
            editedMask = editedMask | app.ROI(ind).createMask();
        end
        app.mask=editedMask;
        app.imf=im2double(app.mask).*im2double(app.im);
        plotout(app)
        end

        % Button pushed function: SaveROIButton
        function SaveROIButtonPushed(app, event)
           roi = findobj(app.UIAxes, 'Type', 'images.roi');
            uisave('roi')            
        end

        % Button pushed function: SaveMaskButton
        function SaveMaskButtonPushed(app, event)
            mask1=app.mask;
            uisave('mask1') 
        end

        % Button pushed function: SaveFinalImageButton
        function SaveFinalImageButtonPushed(app, event)
            imf1=app.imf;
            uisave('imf1')   
        end

        % Button pushed function: LoadROIButton
        function LoadROIButtonPushed(app, event)
          [roifile,roipath]=  uigetfile('*.mat');
             aa =  load([roipath,roifile]);
            roi=aa.roi;
            for ii=1:numel(roi)
           
                type=extractAfter(roi(ii).Type,'roi.');
                switch type
                    case 'polygon'
                        h1 = images.roi.Polygon(app.UIAxes,'Position',roi(ii).Position);
                    case 'rectangle'
                        h2 = images.roi.Rectangle(app.UIAxes,'Position',roi(ii).Position);
                    case  'circle'
                        h3=images.roi.Circle(app.UIAxes,'Center',roi(ii).Center,'Radius',roi(ii).Radius);
                    case 'ellipse'
                        h4=images.roi.Ellipse(app.UIAxes,'Center',roi(ii).Center,'SemiAxes',roi(ii).SemiAxes,'RotationAngle',roi(ii).RotationAngle);
                    case 'freehand'
                        h5=images.roi.Freehand(app.UIAxes,'Position',roi(ii).Position);
                end
            end              
            app.ROI=roi;
            
            UpdateROIButtonPushed(app, event);
            
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [1 1 1];
            app.UIFigure.Position = [100 100 888 469];
            app.UIFigure.Name = 'ROI Editor';

            % Create DONEButton
            app.DONEButton = uibutton(app.UIFigure, 'push');
            app.DONEButton.ButtonPushedFcn = createCallbackFcn(app, @DONEButtonPushed, true);
            app.DONEButton.BackgroundColor = [0.8 0.8 0.8];
            app.DONEButton.FontSize = 18;
            app.DONEButton.FontWeight = 'bold';
            app.DONEButton.FontColor = [0.6353 0.0784 0.1843];
            app.DONEButton.Position = [777 15 100 53];
            app.DONEButton.Text = 'DONE';

            % Create SaveFinalImageButton
            app.SaveFinalImageButton = uibutton(app.UIFigure, 'push');
            app.SaveFinalImageButton.ButtonPushedFcn = createCallbackFcn(app, @SaveFinalImageButtonPushed, true);
            app.SaveFinalImageButton.BackgroundColor = [0.8 0.8 0.8];
            app.SaveFinalImageButton.FontSize = 16;
            app.SaveFinalImageButton.FontWeight = 'bold';
            app.SaveFinalImageButton.FontColor = [0 0 1];
            app.SaveFinalImageButton.Position = [618 15 145 53];
            app.SaveFinalImageButton.Text = 'Save Final Image';

            % Create SaveMaskButton
            app.SaveMaskButton = uibutton(app.UIFigure, 'push');
            app.SaveMaskButton.ButtonPushedFcn = createCallbackFcn(app, @SaveMaskButtonPushed, true);
            app.SaveMaskButton.BackgroundColor = [0.8 0.8 0.8];
            app.SaveMaskButton.FontSize = 16;
            app.SaveMaskButton.FontWeight = 'bold';
            app.SaveMaskButton.FontColor = [0 0 1];
            app.SaveMaskButton.Position = [504 15 101 53];
            app.SaveMaskButton.Text = 'Save Mask';

            % Create SaveROIButton
            app.SaveROIButton = uibutton(app.UIFigure, 'push');
            app.SaveROIButton.ButtonPushedFcn = createCallbackFcn(app, @SaveROIButtonPushed, true);
            app.SaveROIButton.BackgroundColor = [0.8 0.8 0.8];
            app.SaveROIButton.FontSize = 18;
            app.SaveROIButton.FontWeight = 'bold';
            app.SaveROIButton.FontColor = [0 0 1];
            app.SaveROIButton.Position = [395 15 100 53];
            app.SaveROIButton.Text = 'Save ROI';

            % Create FillInsideButton
            app.FillInsideButton = uibutton(app.UIFigure, 'push');
            app.FillInsideButton.ButtonPushedFcn = createCallbackFcn(app, @FillInsideButtonPushed, true);
            app.FillInsideButton.BackgroundColor = [0.8 0.8 0.8];
            app.FillInsideButton.FontSize = 16;
            app.FillInsideButton.FontWeight = 'bold';
            app.FillInsideButton.FontColor = [1 0 0];
            app.FillInsideButton.Position = [272 15 100 53];
            app.FillInsideButton.Text = 'Fill Inside';

            % Create ClearoutsideButton
            app.ClearoutsideButton = uibutton(app.UIFigure, 'push');
            app.ClearoutsideButton.ButtonPushedFcn = createCallbackFcn(app, @ClearoutsideButtonPushed, true);
            app.ClearoutsideButton.BackgroundColor = [0.8 0.8 0.8];
            app.ClearoutsideButton.FontSize = 16;
            app.ClearoutsideButton.FontWeight = 'bold';
            app.ClearoutsideButton.FontColor = [1 0 0];
            app.ClearoutsideButton.Position = [140 15 117 53];
            app.ClearoutsideButton.Text = 'Clear outside';

            % Create ClearinsideButton
            app.ClearinsideButton = uibutton(app.UIFigure, 'push');
            app.ClearinsideButton.ButtonPushedFcn = createCallbackFcn(app, @ClearinsideButtonPushed, true);
            app.ClearinsideButton.BackgroundColor = [0.8 0.8 0.8];
            app.ClearinsideButton.FontSize = 16;
            app.ClearinsideButton.FontWeight = 'bold';
            app.ClearinsideButton.FontColor = [1 0 0];
            app.ClearinsideButton.Position = [11 15 115 53];
            app.ClearinsideButton.Text = 'Clear inside';

            % Create DrawPolygonButton
            app.DrawPolygonButton = uibutton(app.UIFigure, 'push');
            app.DrawPolygonButton.ButtonPushedFcn = createCallbackFcn(app, @DrawPolygonButtonPushed, true);
            app.DrawPolygonButton.BackgroundColor = [0.302 0.7451 0.9333];
            app.DrawPolygonButton.FontWeight = 'bold';
            app.DrawPolygonButton.Position = [11 118 107 38];
            app.DrawPolygonButton.Text = 'Draw Polygon';

            % Create DrawRectangleButton
            app.DrawRectangleButton = uibutton(app.UIFigure, 'push');
            app.DrawRectangleButton.ButtonPushedFcn = createCallbackFcn(app, @DrawRectangleButtonPushed, true);
            app.DrawRectangleButton.BackgroundColor = [0.302 0.7451 0.9333];
            app.DrawRectangleButton.FontWeight = 'bold';
            app.DrawRectangleButton.Position = [11 165 107 38];
            app.DrawRectangleButton.Text = 'Draw Rectangle';

            % Create DrawCircleButton
            app.DrawCircleButton = uibutton(app.UIFigure, 'push');
            app.DrawCircleButton.ButtonPushedFcn = createCallbackFcn(app, @DrawCircleButtonPushed, true);
            app.DrawCircleButton.BackgroundColor = [0.302 0.7451 0.9333];
            app.DrawCircleButton.FontWeight = 'bold';
            app.DrawCircleButton.Position = [11 216 107 38];
            app.DrawCircleButton.Text = 'Draw Circle';

            % Create DrawFreehandButton
            app.DrawFreehandButton = uibutton(app.UIFigure, 'push');
            app.DrawFreehandButton.ButtonPushedFcn = createCallbackFcn(app, @DrawFreehandButtonPushed, true);
            app.DrawFreehandButton.BackgroundColor = [0.302 0.7451 0.9333];
            app.DrawFreehandButton.FontWeight = 'bold';
            app.DrawFreehandButton.Position = [11 259 107 38];
            app.DrawFreehandButton.Text = 'Draw Freehand';

            % Create DrawEllipseButton
            app.DrawEllipseButton = uibutton(app.UIFigure, 'push');
            app.DrawEllipseButton.ButtonPushedFcn = createCallbackFcn(app, @DrawEllipseButtonPushed, true);
            app.DrawEllipseButton.BackgroundColor = [0.302 0.7451 0.9333];
            app.DrawEllipseButton.FontWeight = 'bold';
            app.DrawEllipseButton.Position = [11 305 107 38];
            app.DrawEllipseButton.Text = 'Draw Ellipse';

            % Create UpdateROIButton
            app.UpdateROIButton = uibutton(app.UIFigure, 'push');
            app.UpdateROIButton.ButtonPushedFcn = createCallbackFcn(app, @UpdateROIButtonPushed, true);
            app.UpdateROIButton.BackgroundColor = [0.0745 0.6235 1];
            app.UpdateROIButton.FontSize = 16;
            app.UpdateROIButton.FontWeight = 'bold';
            app.UpdateROIButton.Position = [11 354 107 42];
            app.UpdateROIButton.Text = 'Update ROI';

            % Create LoadROIButton
            app.LoadROIButton = uibutton(app.UIFigure, 'push');
            app.LoadROIButton.ButtonPushedFcn = createCallbackFcn(app, @LoadROIButtonPushed, true);
            app.LoadROIButton.BackgroundColor = [0.0745 0.6235 1];
            app.LoadROIButton.FontSize = 16;
            app.LoadROIButton.FontWeight = 'bold';
            app.LoadROIButton.Position = [11 405 107 36];
            app.LoadROIButton.Text = 'Load ROI';

            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            title(app.UIAxes, 'Input image')
            app.UIAxes.PlotBoxAspectRatio = [1 1 1];
            app.UIAxes.XTick = [];
            app.UIAxes.YTick = [];
            app.UIAxes.Box = 'on';
            app.UIAxes.Position = [120 108 373 341];

            % Create UIAxes2
            app.UIAxes2 = uiaxes(app.UIFigure);
            title(app.UIAxes2, 'Output image')
            zlabel(app.UIAxes2, 'Z')
            app.UIAxes2.PlotBoxAspectRatio = [1 1 1];
            app.UIAxes2.XTick = [];
            app.UIAxes2.YTick = [];
            app.UIAxes2.Box = 'on';
            app.UIAxes2.Position = [504 108 373 341];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';

        end
    end


    end