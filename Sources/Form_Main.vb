
Public Class Form_Main

    Private Sub Form_Main_Load(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.Load
        ' ----------------------------------------------------------------
        'Me.Text = AppTitleAndVersion()  '直接用定义的名字
        ' ---------------------------------------------------------------- 
        Load_INI()
        Menu_Tools_Separator_UpdateChecks()
        Spectrometer_SetRunningModeParams()
        ComboBox_VideoInputDevice_InitWithCurrentDeviceName()
        ComboBox_FileFormat_InitWithCurrentFileFormat()
        SetLocales()
        UpdateUserInterface()
        MakeIrradianceDict()
        ' ---------------------------------------------------------------- CAPTURE
        Capture_INIT(VideoInDevice)
        If VideoCaptureDevice_Is_VFW() Then
            Capture_ConnectFiltersAndRun()
        Else
            Capture_SetVideoFormatParams()
        End If
        ' ---------------------------------------------------------------- DOCK
        If Form_VideoInControls_VisibleAtStart Then
            Form_VideoInControls.Show(Me)
        Else
            Form_VideoInControls.Show(Me)
            Form_VideoInControls.Visible = False
        End If
        DockAllWindows()
        ' ----------------------------------------------------------------
        ToolStrip1.Renderer = New ToolStripButtonRenderer
        ' ---------------------------------------------------------------- Main Timer
        Timer1.Interval = 1
        Timer1.Start()
        ' ---------------------------------------------------------------- Timer 1Hz
        Timer_1Hz.Interval = 1000
        Timer_1Hz.Start()
        ' ----------------------------------------------------------------
        SetDefaultFocus()
        EventsAreEnabled = True
        ' ---------------------------------------------------------------- SHOW
        Refresh()
        Forms_FadeTo(1, 400)
    End Sub

    Private Sub Form1_FormClosing(ByVal sender As Object, ByVal e As System.Windows.Forms.FormClosingEventArgs) Handles Me.FormClosing
        If Not EventsAreEnabled Then Return
        CloseProgram()
    End Sub

    Private Sub CloseProgram()
        Save_INI()
        EventsAreEnabled = False
        Forms_FadeTo(0, 500)
        Me.Refresh()
        Timer1.Stop()
        Capture_STOP()
        Form_VideoInControls.Close()
        Me.Close()
    End Sub

    Private Sub Form1_LocationChanged(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.LocationChanged
        If Not EventsAreEnabled Then Return
        LimitFormPosition(Me)
    End Sub

    Private Sub Form_Main_Resize(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.Resize
        If Not EventsAreEnabled Then Return
        ShowSpectrumGraph()
        DockAllWindows()
    End Sub

    Private Sub Form_Main_Move(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.Move
        If Not EventsAreEnabled Then Return
        DockAllWindows()
    End Sub

    Private Sub SetDefaultFocus()
        PBox_Camera.Focus()
    End Sub

    Friend Sub PictureBox1_Clear()
        PBox_Camera.Image = PBox_Camera.InitialImage
    End Sub

    Friend Sub DockAllWindows()
        Form_VideoInControls.SetSnap()
    End Sub

    Private Sub OpenCloseFormVideoInControls()
        If Form_VideoInControls.Visible Then
            Form_VideoInControls.Visible = False
        Else
            Form_VideoInControls.Visible = True
            Me.Focus()
        End If
        DockAllWindows()
    End Sub

    Private Sub UpdateUserInterface()
        ' ready for language strings not already updated
        Refresh()
    End Sub

    ' ===================================================================================
    '   MenuStrip and ToolStrip Gradients，用渐变色填充菜单栏和工具栏
    ' ===================================================================================
    Private Sub MenuStrip1_Paint(ByVal sender As Object, ByVal e As System.Windows.Forms.PaintEventArgs) Handles MenuStrip1.Paint
        Dim bounds As New Rectangle(0, 0, MenuStrip1.Width, MenuStrip1.Height)
        Dim brush As New Drawing2D.LinearGradientBrush(bounds, Color.FromArgb(230, 230, 230), Color.FromArgb(100, 100, 100), Drawing2D.LinearGradientMode.Horizontal)
        e.Graphics.FillRectangle(brush, bounds)
    End Sub
    Private Sub ToolStrip1_Paint(ByVal sender As System.Object, ByVal e As System.Windows.Forms.PaintEventArgs) Handles ToolStrip1.Paint
        Dim bounds As New Rectangle(0, 0, ToolStrip1.Width, ToolStrip1.Height)
        Dim brush As New Drawing2D.LinearGradientBrush(bounds, Color.White, Color.FromArgb(200, 200, 200), Drawing2D.LinearGradientMode.Vertical)
        e.Graphics.FillRectangle(brush, bounds)
    End Sub

    ' ===================================================================================
    '   ToolStrip PressedButton color，这个地方好像定义的 RUN 按钮
    ' ===================================================================================
    Class ToolStripButtonRenderer
        Inherits System.Windows.Forms.ToolStripProfessionalRenderer
        Protected Overrides Sub OnRenderButtonBackground(ByVal e As ToolStripItemRenderEventArgs)
            Dim btn As ToolStripButton = CType(e.Item, ToolStripButton)
            If btn IsNot Nothing AndAlso btn.CheckOnClick AndAlso btn.Checked Then
                Dim bounds As Rectangle = New Rectangle(0, 0, e.Item.Width - 1, e.Item.Height - 1)
                Dim brush As New Drawing2D.LinearGradientBrush(bounds, Color.Gold, Color.FromArgb(250, 250, 250), Drawing2D.LinearGradientMode.Vertical)
                e.Graphics.FillRectangle(brush, bounds)
                e.Graphics.DrawRectangle(Pens.Orange, bounds)
            Else
                MyBase.OnRenderButtonBackground(e)
            End If
        End Sub
    End Class

    ' ===================================================================================
    '  MenuStrip and ToolStrip accepting the first click
    '  If the form receives a WM_PARENTNOTIFY (528) message and is not focused 
    '  then the form is activated before to exec the message
    ' ===================================================================================
    Protected Overrides Sub WndProc(ByRef m As Message)
        If m.Msg = 528 AndAlso Not Me.Focused Then
            Me.Activate()
        End If
        MyBase.WndProc(m)
    End Sub


    ' =======================================================================================
    '接下来是挨个介绍菜单栏的功能
    '   MENU FILE
    ' =======================================================================================
    Private Sub Menu_Exit_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_File_Exit.Click
        CloseProgram()
    End Sub

    ' =======================================================================================
    '   MENU TOOLS
    ' =======================================================================================
    Private Sub Menu_VideoinControls_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Tools_VideoinControls.Click
        OpenCloseFormVideoInControls()
    End Sub
    Private Sub Menu_Tools_Trim1_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Tools_Trim1.Click
        TrimPoint1 = 436
        TrimPoint2 = 546
        btn_TrimScale.Checked = True
        Spectrometer_SetRunningModeParams()
    End Sub
    Private Sub Menu_Tools_Trim2_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Tools_Trim2.Click
        TrimPoint1 = 436
        TrimPoint2 = 692
        btn_TrimScale.Checked = True
        Spectrometer_SetRunningModeParams()
    End Sub
    Private Sub Menu_Tools_TrimSelect_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Tools_TrimSelect.Click
        Dim s As String
        Dim t As String = "Theremino Spectrometer - " & Msg_TrimTitle
        s = InputBox(Msg_Trim1, t, TrimPoint1.ToString("0.00", Globalization.CultureInfo.InvariantCulture))
        If s <> "" Then TrimPoint1 = CSng(Val(s.Replace(",", ".")))
        If TrimPoint1 < 1 Then TrimPoint1 = 1
        If TrimPoint1 > 2000 Then TrimPoint1 = 2000
        s = InputBox(Msg_Trim2, t, TrimPoint2.ToString("0.00", Globalization.CultureInfo.InvariantCulture))
        If s <> "" Then TrimPoint2 = CSng(Val(s.Replace(",", ".")))
        If TrimPoint2 < 1 Then TrimPoint2 = 1
        If TrimPoint2 > 2000 Then TrimPoint2 = 2000
        If TrimPoint1 > TrimPoint2 Then Swap(TrimPoint1, TrimPoint2)
        btn_TrimScale.Checked = True
        Spectrometer_SetRunningModeParams()
    End Sub

    Private Sub Menu_Tools_SeparatorTab_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Tools_SeparatorTab.Click
        SpectrumFileSeparator = vbTab
        Menu_Tools_Separator_UpdateChecks()
    End Sub
    Private Sub Menu_Tools_SeparatorSemicolon_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Tools_SeparatorSemicolon.Click
        SpectrumFileSeparator = ";"
        Menu_Tools_Separator_UpdateChecks()
    End Sub
    Private Sub Menu_Tools_SeparatorComma_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Tools_SeparatorComma.Click
        SpectrumFileSeparator = ","
        Menu_Tools_Separator_UpdateChecks()
    End Sub
    Private Sub Menu_Tools_Separator_UpdateChecks()
        Menu_Tools_SeparatorTab.Checked = SpectrumFileSeparator = vbTab
        Menu_Tools_SeparatorSemicolon.Checked = SpectrumFileSeparator = ";"
        Menu_Tools_SeparatorComma.Checked = SpectrumFileSeparator = ","
    End Sub

    ' =======================================================================================
    '   MENU LANGUAGE
    ' =======================================================================================
    Private Sub Menu_Language_DropDownOpening(ByVal sender As Object, ByVal e As System.EventArgs) Handles Menu_Language.DropDownOpening
        For Each item As ToolStripMenuItem In Menu_Language.DropDownItems
            If item.Name.EndsWith(Language, StringComparison.InvariantCultureIgnoreCase) Then
                item.Select()
            End If
        Next
    End Sub
    Private Sub Menu_Language_DEU_Click(ByVal sender As Object, ByVal e As System.EventArgs) Handles Menu_Language_DEU.Click
        Language = "DEU"
        SetLocales()
        Save_INI()
        UpdateUserInterface()
    End Sub
    Private Sub Menu_Language_ENG_Click(ByVal sender As Object, ByVal e As System.EventArgs) Handles Menu_Language_ENG.Click
        Language = "ENG"
        SetLocales()
        Save_INI()
        UpdateUserInterface()
    End Sub
    Private Sub Menu_Language_ESP_Click(ByVal sender As Object, ByVal e As System.EventArgs) Handles Menu_Language_ESP.Click
        Language = "ESP"
        SetLocales()
        Save_INI()
        UpdateUserInterface()
    End Sub
    Private Sub Menu_Language_POR_Click(ByVal sender As Object, ByVal e As System.EventArgs) Handles Menu_Language_POR.Click
        Language = "POR"
        SetLocales()
        Save_INI()
        UpdateUserInterface()
    End Sub
    Private Sub Menu_Language_FRA_Click(ByVal sender As Object, ByVal e As System.EventArgs) Handles Menu_Language_FRA.Click
        Language = "FRA"
        SetLocales()
        Save_INI()
        UpdateUserInterface()
    End Sub
    Private Sub Menu_Language_ITA_Click(ByVal sender As Object, ByVal e As System.EventArgs) Handles Menu_Language_ITA.Click
        Language = "ITA"
        SetLocales()
        Save_INI()
        UpdateUserInterface()
    End Sub
    Private Sub Menu_Language_JPN_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Language_JPN.Click
        Language = "JPN"
        SetLocales()
        Save_INI()
        UpdateUserInterface()
    End Sub
    Private Sub Menu_Language_Chinese_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Language_CHI.Click
        Language = "CHI"
        SetLocales()
        Save_INI()
        UpdateUserInterface()
    End Sub

    ' =======================================================================================
    '   MENU HELP
    ' =======================================================================================
    Private Sub Menu_Help_DropDownOpening(ByVal sender As Object, ByVal e As System.EventArgs) Handles Menu_Help.DropDownOpening
        SetLocales()
    End Sub
    Private Sub Menu_Help_ProgramHelp_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Help_ProgramHelp.Click
        OpenLocalizedHelp("Theremino_Spectrometer_Help", ".pdf")
    End Sub
    Private Sub Menu_Help_Technology_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Help_Technology.Click
        OpenLocalizedHelp("Theremino_Spectrometer_Technology", ".pdf")
    End Sub
    Private Sub Menu_Help_Construction_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Help_Construction.Click
        OpenLocalizedHelp("Theremino_Spectrometer_Construction", ".pdf")
    End Sub
    Private Sub Menu_Help_Spectrums_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Help_Spectrums.Click
        OpenLocalizedHelp("Theremino_Spectrometer_Spectrums", ".pdf")
    End Sub
    Private Sub Menu_Help_OpenProgramFolder_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_Help_OpenProgramFolder.Click
        Process.Start(Application.StartupPath)
    End Sub
    Private Sub OpenLocalizedHelp(ByVal name As String, Optional ByVal ext As String = ".rtf")
        Dim fname As String = PlatformAdjustedFileName(Application.StartupPath & "\Docs\" & name & "_" & Language & ext)
        If FileExists(fname) Then
            Process.Start(fname)
        Else
            fname = PlatformAdjustedFileName(Application.StartupPath & "\Docs\" & name & "_ENG" & ext)
            If FileExists(fname) Then
                Process.Start(fname)
            Else
                fname = PlatformAdjustedFileName(Application.StartupPath & "\Docs\" & name & "_ITA" & ext)
                If FileExists(fname) Then
                    Process.Start(fname)
                Else
                    fname = PlatformAdjustedFileName(Application.StartupPath & "\Docs\" & name & ext)
                    If FileExists(fname) Then
                        Process.Start(fname)
                    End If
                End If
            End If
        End If
    End Sub

    ' =======================================================================================
    '   MENU ABOUT
    ' =======================================================================================
    Private Sub Menu_About_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Menu_About.Click
        Form_About.ShowDialog(Me)
    End Sub

    ' =======================================================================================
    '   TOOLSTRIP
    ' =======================================================================================
    Private Sub Tool_VideoInControls_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Tool_VideoControls.Click
        OpenCloseFormVideoInControls()
    End Sub
    Private Sub Tool_SaveSpectrum_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Tools_SaveSpectrum.Click
        Tools_SaveSpectrum.Visible = False
        Tools_SaveSpectrum.Visible = True
        Me.Refresh()
        SaveImage(PBox_Spectrum)
    End Sub
    Private Sub Tools_SaveCamera_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Tools_SaveCamera.Click
        Tools_SaveCamera.Visible = False
        Tools_SaveCamera.Visible = True
        Me.Refresh()
        SaveImage(PBox_Camera)
    End Sub
    Private Sub Tools_SaveTotal_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Tools_SaveTotal.Click
        Tools_SaveTotal.Visible = False
        Tools_SaveTotal.Visible = True
        StatusStrip1.Visible = False
        Me.Refresh()
        SaveImage(Me)
        StatusStrip1.Visible = True
    End Sub
    Private Sub Tools_SpectrumToFile_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Tools_SaveDataFile.Click
        SaveSpectrumToFile()
    End Sub
    Private Sub ToolStrip1_MouseEnter(ByVal sender As Object, ByVal e As System.EventArgs) Handles ToolStrip1.MouseEnter
        Me.Focus() ' with this the toolstrip responds always at the first click
    End Sub

    ' ==============================================================================================================
    '   COMBO BOX - FILE
    ' ==============================================================================================================
    Private Sub ComboBox_FileFormat_InitWithCurrentFileFormat()
        Combo_Init(ComboBox_FileType, FileFormat)
    End Sub
    Private Sub ComboBox_FileFormat_DropDownClosed(ByVal sender As Object, ByVal e As System.EventArgs) Handles ComboBox_FileType.DropDownClosed
        SetDefaultFocus()
    End Sub
    Private Sub ComboBox_FileFormat_SelectedIndexChanged(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles ComboBox_FileType.SelectedIndexChanged
        If Not EventsAreEnabled Then Exit Sub
        FileFormat = Combo_GetValue(ComboBox_FileType)
        SetDefaultFocus()
        Save_INI()
    End Sub
    Private Sub ComboBox_FileFormat_DropDown(ByVal sender As Object, ByVal e As System.EventArgs) Handles ComboBox_FileType.DropDown
        With ComboBox_FileType
            .Items.Clear()
            .Items.Add("JPG")
            .Items.Add("PNG")
            .Items.Add("TIFF")
            .Items.Add("EXIF")
            .Items.Add("EMF")
            .Items.Add("WMF")
            .Items.Add("GIF")
            .Items.Add("BMP")
            Combo_SetIndex_FromString(ComboBox_FileType, FileFormat)
        End With
    End Sub

    ' ==============================================================================================================
    '   COMBO BOX - VIDEO INPUT
    ' ==============================================================================================================
    Private Sub ComboBox_VideoInputDevice_InitWithCurrentDeviceName()
        Combo_Init(ComboBox_VideoInputDevice, VideoInDevice)
    End Sub
    Private Sub ComboBox_VideoInputDevice_DropDownClosed(ByVal sender As Object, ByVal e As System.EventArgs) Handles ComboBox_VideoInputDevice.DropDownClosed
        SetDefaultFocus()
    End Sub
    Private Sub ComboBox_VideoInputDevice_SelectedIndexChanged(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles ComboBox_VideoInputDevice.SelectedIndexChanged
        If Not EventsAreEnabled Then Exit Sub
        VideoInDevice = Combo_GetValue(ComboBox_VideoInputDevice)
        Timer1.Stop()
        Application.DoEvents()
        Capture_STOP()
        Application.DoEvents()
        Capture_INIT(VideoInDevice)
        Capture_ConnectFiltersAndRun()
        Timer1.Start()
        SetDefaultFocus()
        Save_INI()
        If Form_VideoInControls.Visible Then
            Form_VideoInControls.UpdateAllValues()
            Form_VideoInControls.Focus()
        End If
    End Sub
    Private Sub ComboBox_VideoInputDevice_DropDown(ByVal sender As Object, ByVal e As System.EventArgs) Handles ComboBox_VideoInputDevice.DropDown
        Dim fnames() As String
        fnames = EnumFiltersByCategory(FilterCategory.VideoInputDevice)
        With ComboBox_VideoInputDevice
            .Items.Clear()
            For Each fltName As String In fnames
                .Items.Add(fltName)
            Next
            Combo_SetIndex_FromString(ComboBox_VideoInputDevice, VideoInDevice)
        End With
    End Sub

    ' ==============================================================================================================
    '   BUTTONS AND COMMANDS
    ' ==============================================================================================================
    Private Sub txt_FilePath_MouseDoubleClick(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles txt_FilePath.MouseDoubleClick
        SelectSaveFolder()
    End Sub
    Public Sub SelectSaveFolder()
        Dim FBD As System.Windows.Forms.FolderBrowserDialog
        FBD = New System.Windows.Forms.FolderBrowserDialog()
        With FBD
            ' -- USE ROOTFOLDER TO LIMIT USER CHOOSE AREA -- ( No RootFolder NO limits )
            '.RootFolder = Environment.SpecialFolder.MyComputer
            ' --------------------------------------------------------------------------
            .SelectedPath = txt_FilePath.Text
            .Description = vbCr & "Select the ""File Save Path"""
            If .ShowDialog = DialogResult.OK Then
                txt_FilePath.Text = .SelectedPath
            End If
        End With
    End Sub

    Private Sub btn_ResetSpectrumData_ClickButtonArea(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles btn_ResetSpectrumData.ClickButtonArea
        Spectrometer_SetSourceParams()
    End Sub

    ' =======================================================================================
    '   MOUSE CURSOR
    ' =======================================================================================
    Private CursorStartX As Int32
    Private InitialStartX As Int32
    Private InitialEndX As Int32
    Private InitialMinNm As Single
    Private InitialMaxNm As Single
    Private TrimmingPoint As Int32

    Private InitialTP1_X As Single
    Private InitialTP2_X As Single

    Private Sub PBox_Spectrum_MouseDown(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles PBox_Spectrum.MouseDown
        If Not Tools_Run.Checked Then Return
        If e.Button = Windows.Forms.MouseButtons.Left Or e.Button = Windows.Forms.MouseButtons.Right Then
            TrimmingPoint = 0
            If btn_TrimScale.Checked AndAlso e.Y < 15 Then
                'If Math.Abs(e.X - X_From_Nanometers(TrimPoint1)) < 15 Then
                '    TrimmingPoint = 1
                'ElseIf Math.Abs(e.X - X_From_Nanometers(TrimPoint2)) < 15 Then
                '    TrimmingPoint = 2
                'Else
                '    If e.X < PBox_Spectrum.Width / 2 Then
                '        TrimmingPoint = 3
                '    Else
                '        TrimmingPoint = 4
                '    End If
                'End If

                'If e.X < X_From_Nanometers(TrimPoint1 + 20) Then
                '    TrimmingPoint = 1
                'ElseIf e.X > X_From_Nanometers(TrimPoint2 - 20) Then
                '    TrimmingPoint = 2
                'End If


                If e.X < 0.5 * (X_From_Nanometers(TrimPoint1) + X_From_Nanometers(TrimPoint2)) Then
                    TrimmingPoint = 1
                Else
                    TrimmingPoint = 2
                End If


            End If
            CursorStartX = e.X
            InitialStartX = txt_StartX.NumericValueInteger
            InitialEndX = txt_EndX.NumericValueInteger
            InitialMinNm = NanometersMin
            InitialMaxNm = NanometersMax

            InitialTP1_X = X_From_Nanometers(TrimPoint1)
            InitialTP2_X = X_From_Nanometers(TrimPoint2)

        End If
        PBox_Spectrum.Focus()
    End Sub
    Private Sub PBox_Spectrum_MouseMove(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles PBox_Spectrum.MouseMove
        If Not Tools_Run.Checked Then Return
        If e.Button = Windows.Forms.MouseButtons.Left Or e.Button = Windows.Forms.MouseButtons.Right Then
            Dim dx As Int32
            If TrimmingPoint <> 0 Then
                dx = e.X - CursorStartX
                Dim StartX As Single = txt_StartX.NumericValueInteger
                Dim EndX As Single = txt_EndX.NumericValueInteger
                ' --------------------------------------------------------------------- zoom quantity
                Dim zoom As Single
                zoom = 0.2
                zoom = zoom / (InitialTP2_X - InitialTP1_X)
                zoom = zoom * PBox_Spectrum.ClientSize.Width / (EndX - StartX)
                ' ---------------------------------------------------------------------
                Dim err As Single
                Select Case TrimmingPoint
                    Case 1
                        NanometersMin = InitialMinNm - dx * zoom * TrimPoint1

                        'Do
                        '    err = X_From_Nanometers(TrimPoint2) - InitialTP2_X
                        '    NanometersMax += err * 0.01F
                        '    Spectrometer_SetScaleTrimParams()
                        'Loop Until Math.Abs(err) < 0.03 Or NanometersMax > 3999
                        ''Loop Until Math.Abs(err) < GetNmCoeff() / 100 Or NanometersMax > 3999 Or Math.Abs(err) > 1
                        '' Version 2.9 - CHANGED from 0.01 and 1999 to 0.05 and 3999

                        For i As Int32 = 1 To 1000
                            err = X_From_Nanometers(TrimPoint2) - InitialTP2_X
                            NanometersMax += err * 0.01F
                            Spectrometer_SetScaleTrimParams()
                        Next
                        ' Version 3.1 - CHANGED from Loop to For and removed the exit test

                    Case 2
                        NanometersMax = InitialMaxNm - dx * zoom * TrimPoint2

                        'Do
                        '    err = X_From_Nanometers(TrimPoint1) - InitialTP1_X
                        '    NanometersMin += err * 0.01F
                        '    Spectrometer_SetScaleTrimParams()
                        'Loop Until Math.Abs(err) < 0.01 Or NanometersMin < 51

                        For i As Int32 = 1 To 1000
                            err = X_From_Nanometers(TrimPoint1) - InitialTP1_X
                            NanometersMin += err * 0.01F
                            Spectrometer_SetScaleTrimParams()
                        Next
                        ' Version 3.1 - CHANGED from Loop to For and removed the exit test

                End Select

                If NanometersMin > TrimPoint1 - 1 Then NanometersMin = TrimPoint1 - 1
                If NanometersMax < TrimPoint2 + 1 Then NanometersMax = TrimPoint2 + 1

                Spectrometer_SetScaleTrimParams()
                ShowSpectrumGraph()
                PBox_Spectrum.Refresh()
            Else
                dx = ((e.X - CursorStartX) * (InitialEndX - InitialStartX)) \ PBox_Spectrum.ClientSize.Width
                If InitialStartX - dx >= 0 And InitialEndX - dx <= 1000 Then
                    txt_StartX.NumericValueInteger = InitialStartX - dx
                    txt_EndX.NumericValueInteger = InitialEndX - dx
                    Spectrometer_SetSourceParams()
                End If
            End If
        Else
            Dim nm As Int32 = CInt(X_To_Nanometers(e.X))
            Dim h As Int32 = PBox_Spectrum.ClientSize.Height - 15
            Dim perc As Int32
            If ReferenceScale Then
                perc = ((h - e.Y + 15) * 110 \ h)
                If perc > 110 Then perc = 110
            Else
                perc = ((h - e.Y + 15) * 100 \ h)
                If perc > 100 Then perc = 100
            End If
            Label_MaxPeak.Text = "Curs: " & perc & "%  " & nm.ToString & " nm"
        End If
    End Sub
    Private Sub PBox_Spectrum_MouseWheel(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles PBox_Spectrum.MouseWheel
        If Not Tools_Run.Checked Then Return
        Dim StartX As Single = txt_StartX.NumericValueInteger
        Dim EndX As Single = txt_EndX.NumericValueInteger
        ' --------------------------------------------------------------------- zoom quantity
        Dim dx As Single = (e.Delta * EndX - StartX) / 2000.0F
        If Math.Abs(dx) < 1 Then dx = Math.Sign(dx)
        ' --------------------------------------------------------------------- zoom position
        Dim k1 As Single = CSng(e.X / PBox_Spectrum.Width)
        Dim k2 As Single = 1 - k1
        ' --------------------------------------------------------------------- apply the zoom
        txt_StartX.NumericValueInteger = CInt(StartX + dx * k1)
        txt_EndX.NumericValueInteger = CInt(EndX - dx * k2)
        Spectrometer_SetSourceParams()
    End Sub
    Private Sub PBox_Spectrum_MouseEnter(ByVal sender As Object, ByVal e As System.EventArgs) Handles PBox_Spectrum.MouseEnter
        CursorInside = True
    End Sub
    Private Sub PBox_Spectrum_MouseLeave(ByVal sender As Object, ByVal e As System.EventArgs) Handles PBox_Spectrum.MouseLeave
        CursorInside = False
        MaxNanometers_OldValue = -999
    End Sub

    ' ==============================================================================================================
    '   PARAMS
    ' ==============================================================================================================
    Private Sub Params_LostFocus(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles chk_Flip.LostFocus, _
                                                                                                     txt_StartX.LostFocus, _
                                                                                                     txt_StartY.LostFocus, _
                                                                                                     txt_EndX.LostFocus, _
                                                                                                     txt_SizeY.LostFocus, _
                                                                                                     txt_Filter.LostFocus, _
                                                                                                     txt_RisingSpeed.LostFocus, _
                                                                                                     txt_FallingSpeed.LostFocus, _
                                                                                                     txt_FileName.LostFocus
        If Not EventsAreEnabled Then Return
        Save_INI()
    End Sub

    Private Sub Params_Changed(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles chk_Flip.CheckedChanged, _
                                                                                                   txt_StartX.TextChanged, _
                                                                                                   txt_StartY.TextChanged, _
                                                                                                   txt_EndX.TextChanged, _
                                                                                                   txt_SizeY.TextChanged
        If Not EventsAreEnabled Then Return
        txt_StartY.MaxValue = 100 - txt_SizeY.NumericValueInteger
        txt_StartX.MaxValue = txt_EndX.NumericValueInteger - 100
        txt_EndX.MinValue = txt_StartX.NumericValueInteger + 100
        'txt_StartY.MaxValue = 100 - txt_SizeY.NumericValueInteger
        Spectrometer_SetSourceParams()
    End Sub
    Private Sub Filter_Params_Changed(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles txt_Filter.TextChanged, _
                                                                                                          txt_RisingSpeed.TextChanged, _
                                                                                                          txt_FallingSpeed.TextChanged
        If Not EventsAreEnabled Then Return
        Spectrometer_SetRunningModeParams()
    End Sub

    Private Sub btn_Dips_CheckedChanged(ByVal Sender As Object, ByVal e As System.EventArgs) Handles btn_Dips.CheckedChanged
        If Not EventsAreEnabled Then Return
        Spectrometer_SetRunningModeParams()
    End Sub
    Private Sub btn_Peaks_CheckedChanged(ByVal Sender As Object, ByVal e As System.EventArgs) Handles btn_Peaks.CheckedChanged
        If Not EventsAreEnabled Then Return
        Spectrometer_SetRunningModeParams()
    End Sub
    Private Sub btn_Colors_CheckedChanged(ByVal Sender As Object, ByVal e As System.EventArgs) Handles btn_Colors.CheckedChanged
        If Not EventsAreEnabled Then Return
        Spectrometer_SetRunningModeParams()
    End Sub
    Private Sub btn_TrimScale_CheckedChanged(ByVal Sender As Object, ByVal e As System.EventArgs) Handles btn_TrimScale.CheckedChanged
        If Not EventsAreEnabled Then Return
        Spectrometer_SetRunningModeParams()
    End Sub
    Private Sub btn_Reference_CheckedChanged(ByVal Sender As System.Object, ByVal e As System.EventArgs) Handles btn_Reference.CheckedChanged
        If Not EventsAreEnabled Then Return
        If btn_Reference.Checked Then
            Spectrometer_SetReference()
        Else
            Spectrometer_ResetReference()
        End If
    End Sub

    ' ==============================================================================================================
    '   TIMER 1 Hz
    ' ==============================================================================================================
    Private Sub Timer_1Hz_Tick(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Timer_1Hz.Tick
        If Not EventsAreEnabled Then Return
        EnableDisableControls()
    End Sub
    Private Sub EnableDisableControls()
        If FileFormat = "JPG" Then
            Label_JpegQuality.Enabled = True
            txt_JpegQuality.Enabled = True
        Else
            Label_JpegQuality.Enabled = False
            txt_JpegQuality.Enabled = False
        End If
    End Sub

    ' ==============================================================================================================
    '   CAPTURE TIMER
    ' ==============================================================================================================
    Private Timer1_Working As Boolean = False
    Private SlotRunArmed As Boolean = False
    Private SlotStopArmed As Boolean = False
    Private SlotWriteFileArmed As Boolean = False
    Private Sub Timer1_Tick(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Timer1.Tick
        If Not EventsAreEnabled Then Return
        ' -------------------------------------------------------- Slots for RUN / STOP / WRITEFILE
        Dim n As Int32
        n = txt_SlotRun.NumericValueInteger
        If n > 0 Then
            If Slots.ReadSlot_NoNan(n) > 500 Then
                If SlotRunArmed Then
                    SlotRunArmed = False
                    Tools_Run.Checked = True
                End If
            Else
                SlotRunArmed = True
            End If
        End If
        n = txt_SlotStop.NumericValueInteger
        If n > 0 Then
            If Slots.ReadSlot_NoNan(n) > 500 Then
                If SlotStopArmed Then
                    SlotStopArmed = False
                    Tools_Run.Checked = False
                End If
            Else
                SlotStopArmed = True
            End If
        End If
        n = txt_SlotWriteFile.NumericValueInteger
        If n > 0 Then
            If Slots.ReadSlot_NoNan(n) > 500 Then
                If SlotWriteFileArmed Then
                    SlotWriteFileArmed = False
                    SaveSpectrumToFile()
                End If
            Else
                SlotWriteFileArmed = True
            End If
        End If
        ' -------------------------------------------------------- Capture image
        If Not Tools_Run.Checked Then Return
        If Capture_Image Is Nothing OrElse Capture_Image.PixelFormat = Imaging.PixelFormat.Undefined Then
            Capture_NewImageIsReady = False
            Return
        End If
        If Timer1_Working Then Exit Sub
        Timer1_Working = True
        ProcessCapturedImage()
        Timer1_Working = False
    End Sub

    Private Sub txt_FilePath_TextChanged(sender As Object, e As EventArgs) Handles txt_FilePath.TextChanged

    End Sub

    Private Sub PBox_Spectrum_Click(sender As Object, e As EventArgs) Handles PBox_Spectrum.Click

    End Sub
End Class
