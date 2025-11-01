

Public Class Form_VideoInControls

    Private Sub Form_VideoInControls_Load(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.Load
        InitSnap()
    End Sub
    Private Sub Form_VideoInControls_FormClosing(ByVal sender As Object, ByVal e As System.Windows.Forms.FormClosingEventArgs) Handles Me.FormClosing

    End Sub
    Private Sub Form_VideoInControls_Activated(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.Activated
        UpdateAllValues()
    End Sub
    Private Sub SetDefaultFocus()
        btn_Close.Focus()
    End Sub
    Private Sub MyButton_Close_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles btn_Close.ClickButtonArea
        Me.Hide()
    End Sub
    Private Sub MyButton_DefaultAll_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles btn_DefaultAll.ClickButtonArea
        DefaultAllParams()
    End Sub
    Private Sub Tool_SourcePanel_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Tool_OpenSourcePanel.Click
        AsyncExecutionTimer_Start("Source Panel")
    End Sub
    Private Sub Tool_FormatPanel_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Tool_OpenFormatPanel.Click
        AsyncExecutionTimer_Start("Format Panel")
    End Sub


    ' =======================================================================================
    '   SNAP WINDOW
    ' =======================================================================================
    Private SnapPosition As Int32 = 0
    ' --------------------------------------------------------------------- InitSmap must be calle from FormLoad
    Private Sub InitSnap()
        ' ---------------------------------- prepare snap params (timer will be disabled automatically)
        SnapMouseMoveTimer.Enabled = True
        TestSnap()
    End Sub
    ' --------------------------------------------------------------------- Start snap when clicking on TitleBar
    Protected Overrides Sub WndProc(ByRef m As Message)
        Const WM_NCLBUTTONDOWN As Integer = 161
        Const HTCAPTION As Integer = 2
        If m.Msg = WM_NCLBUTTONDOWN AndAlso m.WParam.ToInt32 = HTCAPTION Then
            Me.Select()
            Snap_MouseDown()
            Return
        End If
        MyBase.WndProc(m)
    End Sub
    ' --------------------------------------------------------------------- test snap positions
    Private Sub TestSnap()
        Dim newSnap As Int32 = 0
        If Math.Abs(Me.Left - Form_Main.Right) < 40 And Math.Abs(Me.Top - Form_Main.Top) < 40 Then newSnap = 1
        If Math.Abs(Me.Left + Me.Width - Form_Main.Left) < 40 And Math.Abs(Me.Top - Form_Main.Top) < 40 Then newSnap = 2
        If newSnap <> SnapPosition Then
            SnapPosition = newSnap
            If newSnap <> 0 Then
                SnapMouseMoveTimer.Enabled = False
                SetSnap()
            End If
        End If
    End Sub
    ' --------------------------------------------------------------------- set position to snap positions
    '    must be called from MainForm:DockAllWindows and from ResizeEnd
    ' -----------------------------------------------------------------------------------------------------
    Friend Sub SetSnap()
        If Form_Main.Left < -30000 Then Return
        ' ------------------------------------------------------- large border if Major>5 (not XP)
        Dim bx As Int32 = 0
        Dim by As Int32 = 0
        If Environment.OSVersion.Version.Major > 5 Then
            If Form_Main.FormBorderStyle = Windows.Forms.FormBorderStyle.Sizable Then
                bx = 5
                by = 5
            Else
                bx = 10
                by = 0
            End If
        End If
        ' -------------------------------------------------------
        Select Case SnapPosition
            Case 1 : Me.Left = Form_Main.Right + bx : Me.Top = Form_Main.Top + by
            Case 2 : Me.Left = Form_Main.Left - Me.Size.Width - bx : Me.Top = Form_Main.Top + by
        End Select
    End Sub
    ' --------------------------------------------------------------------- move the window with the mouse
    Private CursorStartPos As Point
    Private FormStartPos As Point
    Private WithEvents SnapMouseMoveTimer As Timer = New Timer

    Private Sub Snap_Form_Move(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.Move
        If Not SnapMouseMoveTimer.Enabled Then Return
        LimitFormPosition(Me)
        TestSnap()
    End Sub
    Private Sub Snap_MouseDown()
        CursorStartPos = Cursor.Position
        FormStartPos = Me.Location
        SnapMouseMoveTimer.Interval = 15
        SnapMouseMoveTimer.Enabled = True
    End Sub
    Private Sub SnapMouseMoveTimer_Tick(ByVal sender As Object, ByVal e As System.EventArgs) Handles SnapMouseMoveTimer.Tick
        If MouseButtonLeftPressed() Then
            Me.Location = New Point(FormStartPos.X + Cursor.Position.X - CursorStartPos.X, _
                                    FormStartPos.Y + Cursor.Position.Y - CursorStartPos.Y)
        Else
            SnapMouseMoveTimer.Enabled = False
        End If
    End Sub


    ' ----------------------------------------------------------------------------------------------
    '  Async Execution Timer  ( used to execute out of event handlers )
    ' ----------------------------------------------------------------------------------------------
    Private WithEvents AsyncExecutionTimer As System.Windows.Forms.Timer = New System.Windows.Forms.Timer
    Private Sub AsyncExecutionTimer_Start(ByVal command As String)
        AsyncExecutionTimer.Tag = command
        AsyncExecutionTimer.Interval = 100
        AsyncExecutionTimer.Start()
    End Sub
    Private Sub Timer_Tick(ByVal sender As Object, ByVal e As System.EventArgs) Handles AsyncExecutionTimer.Tick
        AsyncExecutionTimer.Stop()
        Dim cmd As String = AsyncExecutionTimer.Tag.ToString
        Select Case cmd
            Case "Source Panel"
                ShowCaptureFilterDialog(Me.Handle)
                Capture_GetVideoParams()
                ShowVideoParams()
            Case "Format Panel"
                ShowCapturePinDialog(Me.Handle)
                Capture_GetVideoFormatParams()
                ShowVideoFormatParams()
        End Select
    End Sub

    Friend Sub UpdateAllValues()
        Capture_GetVideoParams()
        ShowVideoParams()
        Capture_GetVideoFormatParams()
        ShowVideoFormatParams()
    End Sub


    ' ==============================================================================================================
    '   COMBO BOX - VideoFormat
    ' ==============================================================================================================
    Private Sub ComboBox_VideoFormat_InitWithCurrentValue()
        Combo_Init(ComboBox_VideoFormat, VideoFormatParams.VideoFormat)
    End Sub
    Private Sub ComboBox_VideoFormat_DropDownClosed(ByVal sender As Object, ByVal e As System.EventArgs) Handles ComboBox_VideoFormat.DropDownClosed
        SetDefaultFocus()
    End Sub
    Private Sub ComboBox_VideoFormat_SelectedIndexChanged(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles ComboBox_VideoFormat.SelectedIndexChanged
        If Not EventsAreEnabled Then Exit Sub
        VideoFormatParams.VideoFormat = Combo_GetValue(ComboBox_VideoFormat)
        '
        Capture_SetVideoFormatParams()
        Capture_GetVideoFormatParams()
        ShowVideoFormatParams()
        SetDefaultFocus()
        Save_INI()
    End Sub
    Private Sub ComboBox_VideoFormat_DropDown(ByVal sender As Object, ByVal e As System.EventArgs) Handles ComboBox_VideoFormat.DropDown
        With ComboBox_VideoFormat
            .Items.Clear()
            .Items.AddRange(Capture_GetValidVideoFormats())
            Combo_SetIndex_FromString(ComboBox_VideoFormat, VideoFormatParams.VideoFormat)
        End With
    End Sub

    ' ==============================================================================================================
    '   COMBO BOX - VideoSize
    ' ==============================================================================================================
    Private Sub ComboBox_VideoSize_InitWithCurrentValue()
        Combo_Init(ComboBox_VideoSize, VideoFormatParams.VideoSize)
    End Sub
    Private Sub ComboBox_VideoSize_DropDownClosed(ByVal sender As Object, ByVal e As System.EventArgs) Handles ComboBox_VideoSize.DropDownClosed
        SetDefaultFocus()
    End Sub
    Private Sub ComboBox_VideoSize_SelectedIndexChanged(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles ComboBox_VideoSize.SelectedIndexChanged
        If Not EventsAreEnabled Then Exit Sub
        VideoFormatParams.VideoSize = Combo_GetValue(ComboBox_VideoSize)
        '
        Capture_SetVideoFormatParams()
        Capture_GetVideoFormatParams()
        ShowVideoFormatParams()
        SetDefaultFocus()
        Save_INI()
    End Sub
    Private Sub ComboBox_VideoSize_DropDown(ByVal sender As Object, ByVal e As System.EventArgs) Handles ComboBox_VideoSize.DropDown
        With ComboBox_VideoSize
            .Items.Clear()
            .Items.AddRange(Capture_GetValidVideoSizes())
            Combo_SetIndex_FromString(ComboBox_VideoSize, VideoFormatParams.VideoSize)
        End With
    End Sub


    ' ==============================================================================================================
    '   COMBO BOX - VideoFPS
    ' ==============================================================================================================
    Private Sub ComboBox_VideoFPS_InitWithCurrentValue()
        Combo_Init(ComboBox_VideoFPS, VideoFormatParams.VideoFPS)
    End Sub
    Private Sub ComboBox_VideoFPS_DropDownClosed(ByVal sender As Object, ByVal e As System.EventArgs) Handles ComboBox_VideoFPS.DropDownClosed
        SetDefaultFocus()
    End Sub
    Private Sub ComboBox_VideoFPS_SelectedIndexChanged(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles ComboBox_VideoFPS.SelectedIndexChanged
        If Not EventsAreEnabled Then Exit Sub
        VideoFormatParams.VideoFPS = Combo_GetValue(ComboBox_VideoFPS)
        '
        Capture_SetVideoFormatParams()
        Capture_GetVideoFormatParams()
        ShowVideoFormatParams()
        SetDefaultFocus()
        Save_INI()
    End Sub
    Private Sub ComboBox_VideoInputDevice_DropDown(ByVal sender As Object, ByVal e As System.EventArgs) Handles ComboBox_VideoFPS.DropDown
        With ComboBox_VideoFPS
            .Items.Clear()
            .Items.AddRange(Capture_GetValidVideoFPS())
            Combo_SetIndex_FromString(ComboBox_VideoFPS, VideoFormatParams.VideoFPS)
        End With
    End Sub

    Private Yposition As Int32
    Private Sub ShowVideoParams()
        Dim OldEventsEnabled As Boolean = EventsAreEnabled
        EventsAreEnabled = False
        Yposition = Label_Default.Top + 18
        InitTrackBarAndLabel(MyButton_Exposure, CheckBox_Exposure, TrackBar_Exposure, Label_Exposure, VideoInputParams.Exposure)
        InitTrackBarAndLabel(MyButton_Gain, CheckBox_Gain, TrackBar_Gain, Label_Gain, VideoInputParams.Gain)
        InitTrackBarAndLabel(MyButton_Brightness, CheckBox_Brightness, TrackBar_Brightness, Label_Brightness, VideoInputParams.Brightness)
        InitTrackBarAndLabel(MyButton_Contrast, CheckBox_Contrast, TrackBar_Contrast, Label_Contrast, VideoInputParams.Contrast)
        InitTrackBarAndLabel(MyButton_Gamma, CheckBox_Gamma, TrackBar_Gamma, Label_Gamma, VideoInputParams.Gamma)
        InitTrackBarAndLabel(MyButton_BackLight, CheckBox_BackLight, TrackBar_BackLight, Label_BackLight, VideoInputParams.BackLight)
        Yposition += 4
        InitTrackBarAndLabel(MyButton_Saturation, CheckBox_Saturation, TrackBar_Saturation, Label_Saturation, VideoInputParams.Saturation)
        InitTrackBarAndLabel(MyButton_WhiteBalance, CheckBox_WhiteBalance, TrackBar_WhiteBalance, Label_WhiteBalance, VideoInputParams.WhiteBalance)
        InitTrackBarAndLabel(MyButton_Hue, CheckBox_Hue, TrackBar_Hue, Label_Hue, VideoInputParams.Hue)
        InitTrackBarAndLabel(MyButton_ColorEnable, CheckBox_ColorEnable, TrackBar_ColorEnable, Label_ColorEnable, VideoInputParams.ColorEnable)
        Yposition += 4
        InitTrackBarAndLabel(MyButton_Zoom, CheckBox_Zoom, TrackBar_Zoom, Label_Zoom, VideoInputParams.Zoom)
        InitTrackBarAndLabel(MyButton_Pan, CheckBox_Pan, TrackBar_Pan, Label_Pan, VideoInputParams.Pan)
        InitTrackBarAndLabel(MyButton_Tilt, CheckBox_Tilt, TrackBar_Tilt, Label_Tilt, VideoInputParams.Tilt)
        Yposition += 4
        InitTrackBarAndLabel(MyButton_Sharpness, CheckBox_Sharpness, TrackBar_Sharpness, Label_Sharpness, VideoInputParams.Sharpness)
        InitTrackBarAndLabel(MyButton_Focus, CheckBox_Focus, TrackBar_Focus, Label_Focus, VideoInputParams.Focus)
        EventsAreEnabled = OldEventsEnabled
    End Sub

    Private Sub ShowVideoFormatParams()
        If VideoCaptureDevice_Is_VFW() Then
            Label_Compression.Visible = False
            Label_VideoSize.Visible = False
            Label_MaxFps.Visible = False
            Label_Default.Visible = False
            Label_Auto.Visible = False
            ComboBox_VideoFormat.Visible = False
            ComboBox_VideoSize.Visible = False
            ComboBox_VideoFPS.Visible = False
            Combo_Init(ComboBox_VideoFormat, "")
            Combo_Init(ComboBox_VideoSize, "")
            Combo_Init(ComboBox_VideoFPS, "")
            btn_DefaultAll.Visible = False
            btn_Close.Top = 50
            Me.Height = 110
        Else
            Label_Compression.Visible = True
            Label_VideoSize.Visible = True
            Label_MaxFps.Visible = True
            Label_Default.Visible = True
            Label_Auto.Visible = True
            ComboBox_VideoFormat.Visible = True
            ComboBox_VideoSize.Visible = True
            ComboBox_VideoFPS.Visible = True
            ComboBox_VideoFormat_InitWithCurrentValue()
            ComboBox_VideoSize_InitWithCurrentValue()
            ComboBox_VideoFPS_InitWithCurrentValue()
            btn_DefaultAll.Top = Yposition + 8
            btn_DefaultAll.Visible = True
            btn_Close.Top = Yposition + 10
            Me.Height = Yposition + 72
        End If
    End Sub


    Private Sub InitTrackBarAndLabel(ByVal button As MyButton, ByVal chkbox As CheckBox, ByVal tbar As TrackBar, ByVal lbl As Label, ByVal param As Capture_VideoInputParam)

        button.Top = Yposition
        chkbox.Top = Yposition + 7
        tbar.Top = Yposition - 8
        lbl.Top = Yposition + 1

        If param._AutoValid Then
            chkbox.Visible = True
            chkbox.Checked = param._AutoChecked             '前值param._AutoChecked，这个地方起效果了，取消选中了
            If param._AutoChecked Then
                tbar.Enabled = False '前值False,改成True之后，滑动条确实是解锁了，但没有办法控制
                lbl.Enabled = False  '前值False
            Else
                tbar.Enabled = True
                lbl.Enabled = True
            End If
        Else
            chkbox.Visible = False
            tbar.Enabled = True
            lbl.Enabled = True
        End If

        'If param._Max <> param._Min Then
        If param._ManualValid Then
            button.Visible = True
            tbar.Visible = True
            lbl.Visible = True
            If param._Min < param._Max Then
                tbar.Minimum = param._Min
                tbar.Maximum = param._Max
            Else
                tbar.Minimum = param._Max
                tbar.Maximum = param._Min
            End If
            tbar.SmallChange = param._Step
            tbar.TickFrequency = CInt(Math.Abs(param._Max - param._Min) / 10)
            If param._Value > param._Max Then param._Value = param._Max
            If param._Value < param._Min Then param._Value = param._Min
            tbar.Value = param._Value
            lbl.Text = param._Value.ToString
            Yposition += 27
        Else
            button.Visible = False
            chkbox.Visible = False
            tbar.Visible = False
            lbl.Visible = False
            'lbl.Text = ""
        End If
    End Sub


    Private Sub Auto_CheckBoxes_CheckedChanged(ByVal sender As Object, ByVal e As System.EventArgs) Handles CheckBox_Exposure.CheckedChanged, _
                                                                                                            CheckBox_Gain.CheckedChanged, _
                                                                                                            CheckBox_Brightness.CheckedChanged, _
                                                                                                            CheckBox_Contrast.CheckedChanged, _
                                                                                                            CheckBox_Gamma.CheckedChanged, _
                                                                                                            CheckBox_BackLight.CheckedChanged, _
                                                                                                            CheckBox_Saturation.CheckedChanged, _
                                                                                                            CheckBox_WhiteBalance.CheckedChanged, _
                                                                                                            CheckBox_Hue.CheckedChanged, _
                                                                                                            CheckBox_ColorEnable.CheckedChanged, _
                                                                                                            CheckBox_Zoom.CheckedChanged, _
                                                                                                            CheckBox_Pan.CheckedChanged, _
                                                                                                            CheckBox_Tilt.CheckedChanged, _
                                                                                                            CheckBox_Sharpness.CheckedChanged, _
                                                                                                            CheckBox_Focus.CheckedChanged
        If Not EventsAreEnabled Then Return
        SetAllAuto()
        ShowVideoParams()
    End Sub

    Private Sub TrackBars_MouseLeave(ByVal sender As Object, ByVal e As System.EventArgs) Handles TrackBar_Exposure.MouseLeave, _
                                                                                                  TrackBar_Gain.MouseLeave, _
                                                                                                  TrackBar_Brightness.MouseLeave, _
                                                                                                  TrackBar_Contrast.MouseLeave, _
                                                                                                  TrackBar_Gamma.MouseLeave, _
                                                                                                  TrackBar_BackLight.MouseLeave, _
                                                                                                  TrackBar_Saturation.MouseLeave, _
                                                                                                  TrackBar_WhiteBalance.MouseLeave, _
                                                                                                  TrackBar_Hue.MouseLeave, _
                                                                                                  TrackBar_ColorEnable.MouseLeave, _
                                                                                                  TrackBar_Zoom.MouseLeave, _
                                                                                                  TrackBar_Pan.MouseLeave, _
                                                                                                  TrackBar_Tilt.MouseLeave, _
                                                                                                  TrackBar_Sharpness.MouseLeave, _
                                                                                                  TrackBar_Focus.MouseLeave
        If Not EventsAreEnabled Then Return
        SetDefaultFocus()
    End Sub

    Private Sub TrackBar_Exposure_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Exposure.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Exposure._Value = TrackBar_Exposure.Value
        Capture_SetCameraControlParam(CameraControlProperty.Exposure, VideoInputParams.Exposure)
        Label_Exposure.Text = TrackBar_Exposure.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Gain_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Gain.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Gain._Value = TrackBar_Gain.Value
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Gain, VideoInputParams.Gain)
        Label_Gain.Text = TrackBar_Gain.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Brightness_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Brightness.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Brightness._Value = TrackBar_Brightness.Value
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Brightness, VideoInputParams.Brightness)
        Label_Brightness.Text = TrackBar_Brightness.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Contrast_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Contrast.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Contrast._Value = TrackBar_Contrast.Value
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Contrast, VideoInputParams.Contrast)
        Label_Contrast.Text = TrackBar_Contrast.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Gamma_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Gamma.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Gamma._Value = TrackBar_Gamma.Value
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Gamma, VideoInputParams.Gamma)
        Label_Gamma.Text = TrackBar_Gamma.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_BackLight_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_BackLight.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.BackLight._Value = TrackBar_BackLight.Value
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.BacklightCompensation, VideoInputParams.BackLight)
        Label_BackLight.Text = TrackBar_BackLight.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Saturation_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Saturation.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Saturation._Value = TrackBar_Saturation.Value
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Saturation, VideoInputParams.Saturation)
        Label_Saturation.Text = TrackBar_Saturation.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_WhiteBalance_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_WhiteBalance.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.WhiteBalance._Value = TrackBar_WhiteBalance.Value
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.WhiteBalance, VideoInputParams.WhiteBalance)
        Label_WhiteBalance.Text = TrackBar_WhiteBalance.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Hue_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Hue.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Hue._Value = TrackBar_Hue.Value
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Hue, VideoInputParams.Hue)
        Label_Hue.Text = TrackBar_Hue.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_ColorEnable_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_ColorEnable.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.ColorEnable._Value = TrackBar_ColorEnable.Value
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.ColorEnable, VideoInputParams.ColorEnable)
        Label_ColorEnable.Text = TrackBar_ColorEnable.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Zoom_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Zoom.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Zoom._Value = TrackBar_Zoom.Value
        Capture_SetCameraControlParam(CameraControlProperty.Zoom, VideoInputParams.Zoom)
        Label_Zoom.Text = TrackBar_Zoom.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Pan_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Pan.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Pan._Value = TrackBar_Pan.Value
        Capture_SetCameraControlParam(CameraControlProperty.Pan, VideoInputParams.Pan)
        Label_Pan.Text = TrackBar_Pan.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Tilt_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Tilt.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Tilt._Value = TrackBar_Tilt.Value
        Capture_SetCameraControlParam(CameraControlProperty.Tilt, VideoInputParams.Tilt)
        Label_Tilt.Text = TrackBar_Tilt.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Sharpness_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Sharpness.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Sharpness._Value = TrackBar_Sharpness.Value
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Sharpness, VideoInputParams.Sharpness)
        Label_Sharpness.Text = TrackBar_Sharpness.Value.ToString
        'Application.DoEvents()
    End Sub
    Private Sub TrackBar_Focus_Scroll(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TrackBar_Focus.Scroll
        If Not EventsAreEnabled Then Return
        VideoInputParams.Focus._Value = TrackBar_Focus.Value
        Capture_SetCameraControlParam(CameraControlProperty.Focus, VideoInputParams.Focus)
        Label_Focus.Text = TrackBar_Focus.Value.ToString
        'Application.DoEvents()
    End Sub

    Private Sub SetAllAuto()
        VideoInputParams.Exposure._AutoChecked = CheckBox_Exposure.Checked      'Gets a value indicating whether the CheckBox is in the checked state.
        VideoInputParams.Gain._AutoChecked = CheckBox_Gain.Checked
        VideoInputParams.Brightness._AutoChecked = CheckBox_Brightness.Checked
        VideoInputParams.Contrast._AutoChecked = CheckBox_Contrast.Checked
        VideoInputParams.Gamma._AutoChecked = CheckBox_Gamma.Checked
        VideoInputParams.BackLight._AutoChecked = CheckBox_BackLight.Checked
        VideoInputParams.Saturation._AutoChecked = CheckBox_Saturation.Checked
        VideoInputParams.WhiteBalance._AutoChecked = CheckBox_WhiteBalance.Checked
        VideoInputParams.Hue._AutoChecked = CheckBox_Hue.Checked
        VideoInputParams.ColorEnable._AutoChecked = CheckBox_ColorEnable.Checked
        VideoInputParams.Zoom._AutoChecked = CheckBox_Zoom.Checked
        VideoInputParams.Pan._AutoChecked = CheckBox_Pan.Checked
        VideoInputParams.Tilt._AutoChecked = CheckBox_Tilt.Checked
        VideoInputParams.Sharpness._AutoChecked = CheckBox_Sharpness.Checked
        VideoInputParams.Focus._AutoChecked = CheckBox_Focus.Checked
        Capture_SetVideoParams()
    End Sub

    Private Sub SetAllParams()
        VideoInputParams.Exposure._Value = TrackBar_Exposure.Value
        VideoInputParams.Gain._Value = TrackBar_Gain.Value
        VideoInputParams.Brightness._Value = TrackBar_Brightness.Value
        VideoInputParams.Contrast._Value = TrackBar_Contrast.Value
        VideoInputParams.Gamma._Value = TrackBar_Gamma.Value
        VideoInputParams.BackLight._Value = TrackBar_BackLight.Value
        VideoInputParams.Saturation._Value = TrackBar_Saturation.Value
        VideoInputParams.WhiteBalance._Value = TrackBar_WhiteBalance.Value
        VideoInputParams.Hue._Value = TrackBar_Hue.Value
        VideoInputParams.ColorEnable._Value = TrackBar_ColorEnable.Value
        VideoInputParams.Zoom._Value = TrackBar_Zoom.Value
        VideoInputParams.Pan._Value = TrackBar_Pan.Value
        VideoInputParams.Tilt._Value = TrackBar_Tilt.Value
        VideoInputParams.Sharpness._Value = TrackBar_Sharpness.Value
        VideoInputParams.Focus._Value = TrackBar_Focus.Value
        Capture_SetVideoParams()
    End Sub

    Private Sub DefaultAllParams()
        VideoInputParams.Exposure._Value = VideoInputParams.Exposure._Default
        VideoInputParams.Gain._Value = VideoInputParams.Gain._Default
        VideoInputParams.Brightness._Value = VideoInputParams.Brightness._Default
        VideoInputParams.Contrast._Value = VideoInputParams.Contrast._Default
        VideoInputParams.Gamma._Value = VideoInputParams.Gamma._Default
        VideoInputParams.BackLight._Value = VideoInputParams.BackLight._Default
        VideoInputParams.Saturation._Value = VideoInputParams.Saturation._Default
        VideoInputParams.WhiteBalance._Value = VideoInputParams.WhiteBalance._Default
        VideoInputParams.Hue._Value = VideoInputParams.Hue._Default
        VideoInputParams.ColorEnable._Value = VideoInputParams.ColorEnable._Default
        VideoInputParams.Zoom._Value = VideoInputParams.Zoom._Default
        VideoInputParams.Pan._Value = VideoInputParams.Pan._Default
        VideoInputParams.Tilt._Value = VideoInputParams.Tilt._Default
        VideoInputParams.Sharpness._Value = VideoInputParams.Sharpness._Default
        VideoInputParams.Focus._Value = VideoInputParams.Focus._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub


    Private Sub MyButton_Exposure_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Exposure.ClickButtonArea
        VideoInputParams.Exposure._Value = VideoInputParams.Exposure._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Gain_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Gain.ClickButtonArea
        VideoInputParams.Gain._Value = VideoInputParams.Gain._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Brightness_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Brightness.ClickButtonArea
        VideoInputParams.Brightness._Value = VideoInputParams.Brightness._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Contrast_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Contrast.ClickButtonArea
        VideoInputParams.Contrast._Value = VideoInputParams.Contrast._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Gamma_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Gamma.ClickButtonArea
        VideoInputParams.Gamma._Value = VideoInputParams.Gamma._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_BackLight_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_BackLight.ClickButtonArea
        VideoInputParams.BackLight._Value = VideoInputParams.BackLight._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Saturation_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Saturation.ClickButtonArea
        VideoInputParams.Saturation._Value = VideoInputParams.Saturation._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_WhiteBalance_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_WhiteBalance.ClickButtonArea
        VideoInputParams.WhiteBalance._Value = VideoInputParams.WhiteBalance._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Hue_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Hue.ClickButtonArea
        VideoInputParams.Hue._Value = VideoInputParams.Hue._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_ColorEnable_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_ColorEnable.ClickButtonArea
        VideoInputParams.ColorEnable._Value = VideoInputParams.ColorEnable._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Zoom_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Zoom.ClickButtonArea
        VideoInputParams.Zoom._Value = VideoInputParams.Zoom._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Pan_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Pan.ClickButtonArea
        VideoInputParams.Pan._Value = VideoInputParams.Pan._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Tilt_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Tilt.ClickButtonArea
        VideoInputParams.Tilt._Value = VideoInputParams.Tilt._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Sharpness_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Sharpness.ClickButtonArea
        VideoInputParams.Sharpness._Value = VideoInputParams.Sharpness._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub
    Private Sub MyButton_Focus_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles MyButton_Focus.ClickButtonArea
        VideoInputParams.Focus._Value = VideoInputParams.Focus._Default
        Capture_SetVideoParams()
        ShowVideoParams()
    End Sub


    Friend Sub Capture_GetVideoParams()
        'Dim t As PrecisionTimer = New PrecisionTimer
        '
        ' --------------------------------------------------------------------------------------------------
        ' existing params tests
        ' --------------------------------------------------------------------------------------------------
        'Dim p As Capture_VideoInputParam
        'p = Capture_GetCameraControlParam(CameraControlProperty.Iris)
        'p = Capture_GetCameraControlParam(CameraControlProperty.Roll)
        ' --------------------------------------------------------------------------------------------------
        '
        VideoInputParams.Exposure = Capture_GetCameraControlParam(CameraControlProperty.Exposure)
        VideoInputParams.Gain = Capture_GetVideoProcAmpParam(VideoProcAmpProperty.Gain)
        VideoInputParams.Brightness = Capture_GetVideoProcAmpParam(VideoProcAmpProperty.Brightness)
        VideoInputParams.Contrast = Capture_GetVideoProcAmpParam(VideoProcAmpProperty.Contrast)
        VideoInputParams.Gamma = Capture_GetVideoProcAmpParam(VideoProcAmpProperty.Gamma)
        VideoInputParams.BackLight = Capture_GetVideoProcAmpParam(VideoProcAmpProperty.BacklightCompensation)
        '
        VideoInputParams.Saturation = Capture_GetVideoProcAmpParam(VideoProcAmpProperty.Saturation)
        VideoInputParams.WhiteBalance = Capture_GetVideoProcAmpParam(VideoProcAmpProperty.WhiteBalance)
        VideoInputParams.Hue = Capture_GetVideoProcAmpParam(VideoProcAmpProperty.Hue)
        VideoInputParams.ColorEnable = Capture_GetVideoProcAmpParam(VideoProcAmpProperty.ColorEnable)
        '
        VideoInputParams.Zoom = Capture_GetCameraControlParam(CameraControlProperty.Zoom)
        VideoInputParams.Pan = Capture_GetCameraControlParam(CameraControlProperty.Pan)
        VideoInputParams.Tilt = Capture_GetCameraControlParam(CameraControlProperty.Tilt)
        '
        VideoInputParams.Sharpness = Capture_GetVideoProcAmpParam(VideoProcAmpProperty.Sharpness)
        VideoInputParams.Focus = Capture_GetCameraControlParam(CameraControlProperty.Focus)
        '
        ' ------------------------------------------------------ test 
        '
        't.DebugPrintMicrosec()
        'Debug.Print("READ: " & VideoInputParams.Exposure._Value.ToString)
        'Debug.Print("READ: " & VideoInputParams.Brightness._Value.ToString)
    End Sub

    Friend Sub Capture_SetVideoParams()
        Capture_SetCameraControlParam(CameraControlProperty.Exposure, VideoInputParams.Exposure)
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Gain, VideoInputParams.Gain)
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Brightness, VideoInputParams.Brightness)
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Contrast, VideoInputParams.Contrast)
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Gamma, VideoInputParams.Gamma)
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.BacklightCompensation, VideoInputParams.BackLight)
        '
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Saturation, VideoInputParams.Saturation)
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.WhiteBalance, VideoInputParams.WhiteBalance)
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Hue, VideoInputParams.Hue)
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.ColorEnable, VideoInputParams.ColorEnable)
        '
        Capture_SetCameraControlParam(CameraControlProperty.Zoom, VideoInputParams.Zoom)
        Capture_SetCameraControlParam(CameraControlProperty.Pan, VideoInputParams.Pan)
        Capture_SetCameraControlParam(CameraControlProperty.Tilt, VideoInputParams.Tilt)
        '
        Capture_SetVideoProcAmpParam(VideoProcAmpProperty.Sharpness, VideoInputParams.Sharpness)
        Capture_SetCameraControlParam(CameraControlProperty.Focus, VideoInputParams.Focus)
        '
        'Debug.Print("SET: " & VideoInputParams.Exposure._Value.ToString)
    End Sub

End Class