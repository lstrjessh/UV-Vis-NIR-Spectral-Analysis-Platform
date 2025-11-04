
Imports System.Drawing.Imaging

Module Module_SaveLoad

    Friend VideoInDevice As String = ""
    Friend FileFormat As String = ""

    Friend GCI As Globalization.CultureInfo = Globalization.CultureInfo.InvariantCulture

    ' =======================================================================================================
    '   APP TITLE AND VERSION
    ' =======================================================================================================
    Friend Function AppTitleAndVersion(Optional ByVal Title As String = "") As String
        If Title = "" Then Title = Replace(My.Application.Info.AssemblyName, "_", " ")
        Dim s() As String = Split(My.Application.Info.Version.ToString, ".")
        Return Title & " - V" & s(0) & "." & s(1)
    End Function

    ' =======================================================================================
    '  FORM FUNCTIONS
    ' =======================================================================================
    Friend Sub LimitFormPosition(ByVal f As System.Windows.Forms.Form)
        If f.WindowState <> FormWindowState.Normal Then Return
        GetMaxScreenBounds()
        EnsureFormVisible(f)
        'EnsureFormCompletelyVisible(f)
    End Sub

    Private SB As Rectangle = New Rectangle(Integer.MaxValue, Integer.MaxValue, Integer.MinValue, Integer.MinValue)

    Private Sub GetMaxScreenBounds()
        For Each s As Screen In System.Windows.Forms.Screen.AllScreens
            SB = Rectangle.Union(SB, s.WorkingArea)
        Next
    End Sub

    'Private Sub EnsureFormCompletelyVisible(ByVal frm As Form)
    '    With frm
    '        .Width = Math.Min(.Width, SB.Width)         ' not more than a maximized window
    '        .Height = Math.Min(.Height, SB.Height)      ' not more than a maximized window
    '        .Width = Math.Max(.Width, 32)               ' at least 32x24
    '        .Height = Math.Max(.Height, 24)             ' at least 32x24
    '        .Left = Math.Min(.Left, SB.Right - .Width)  ' not beyond the right border
    '        .Top = Math.Min(.Top, SB.Bottom - .Height)  ' not beyond the bottom border
    '        .Left = Math.Max(.Left, SB.Left)            ' at least at the left border
    '        .Top = Math.Max(.Top, SB.Top)               ' at least at the top border
    '    End With
    'End Sub

    Private Sub EnsureFormVisible(ByVal frm As Form)
        With frm
            .Width = Math.Min(.Width, SB.Width)             ' not more than VIRTUALSCREEN dimensions
            .Height = Math.Min(.Height, SB.Height)          ' not more than VIRTUALSCREEN dimensions 
            .Width = Math.Max(.Width, 32)                   ' at least 32x24
            .Height = Math.Max(.Height, 24)                 ' at least 32x24
            .Left = Math.Min(.Left, SB.Right - 50)          ' not beyond right border - 50 pixels
            .Top = Math.Min(.Top, SB.Bottom - 100)          ' not beyond bottom border - 50 pixels
            .Left = Math.Max(.Left, SB.Left + 100 - .Width) ' at least at left border + 50 pixels
            .Top = Math.Max(.Top, SB.Top - 10)              ' at least at top border
        End With
    End Sub

    ' (The value of the RestoreBounds property is valid only 
    '   when the WindowState property of the Form class is not equal to Normal)
    Friend Function GetFormRectangle(ByVal frm As Form) As Rectangle
        Dim r As Rectangle
        If frm.WindowState = FormWindowState.Normal Then
            r = frm.Bounds
        Else
            r = frm.RestoreBounds
        End If
        Return r
    End Function


    ' ================================================================================================
    '  Private Read-Write functions
    ' ================================================================================================
    Private Function TabString(ByVal Name As String, _
                               Optional ByVal Value As Double = Double.NaN, _
                               Optional ByVal fmt As String = "") As String

        Dim nTab As Int32 = Math.Max(0, 22 - Name.Length)
        If Double.IsNaN(Value) Then
            Return Name
        Else
            Return Name & "=" & Strings.StrDup(nTab, " ") & Value.ToString(fmt, GCI)
        End If
    End Function
    Private Function TabString(ByVal Name As String, _
                               ByVal Value As Boolean) As String

        Dim nTab As Int32 = Math.Max(0, 22 - Name.Length)

        Return Name & "=" & Strings.StrDup(nTab, " ") & Value.ToString
    End Function
    Private Function TabString(ByVal Name As String, _
                               ByVal Value As String) As String

        Dim nTab As Int32 = Math.Max(0, 22 - Name.Length)

        Return Name & "=" & Strings.StrDup(nTab, " ") & Value
    End Function
    Private Function Val_Int(ByVal l As String) As Int32
        Return CInt(Val(l))
    End Function
    Private Function Val_Double(ByVal l As String) As Double
        Return Val(l.Replace(",", "."))
    End Function
    Private Function Val_Single(ByVal l As String) As Single
        Return CSng(Val(l.Replace(",", ".")))
    End Function
    Friend Function ExtractParamName(ByRef s As String) As String
        ' ------------------------- Returns the first field from begin to the first "=" symbol
        ' -------------------------  and removes it from the string
        Dim i As Integer
        i = InStr(s, "=")
        If i > 0 Then
            ExtractParamName = Trim(Strings.Left(s, i - 1))
            s = Trim(Mid(s, i + 1))
        Else
            ExtractParamName = Trim(s)
            s = ""
        End If
    End Function
    Private Function AssemblyName() As String
        Return System.Reflection.Assembly.GetExecutingAssembly.GetName.Name
    End Function


    ' ==================================================================================================
    '  SAVE LOAD -- Program INI
    ' ==================================================================================================
    Friend Sub Save_INI()
        Dim iniFileName As String = Application.StartupPath & "\" & AssemblyName() & "_INI.txt"
        Dim f As System.IO.StreamWriter = Nothing
        Try
            f = IO.File.CreateText(iniFileName)
            '
            f.WriteLine(" Program Params")
            f.WriteLine("===========================================")
            '
            f.WriteLine(" Program Params")
            f.WriteLine("===========================================")
            Dim r As Rectangle
            r = GetFormRectangle(Form_Main)
            f.WriteLine(TabString("FormMain_Top", r.Top))
            f.WriteLine(TabString("FormMain_Left", r.Left))
            f.WriteLine(TabString("FormMain_Width", r.Width))
            f.WriteLine(TabString("FormMain_Height", r.Height))
            f.WriteLine(TabString("FormMain_WindowState", Form_Main.WindowState))
            '
            r = GetFormRectangle(Form_VideoInControls)
            f.WriteLine(TabString("Form_VideoInControls_Top", r.Top))
            f.WriteLine(TabString("Form_VideoInControls_Left", r.Left))
            f.WriteLine(TabString("Form_VideoInControls_VisibleAtStart", Form_VideoInControls.Visible And Form_VideoInControls.WindowState <> FormWindowState.Minimized))
            '
            f.WriteLine("")
            f.WriteLine(TabString("==========================================="))
            f.WriteLine(TabString("Language", Language))
            '
            f.WriteLine(TabString(""))
            f.WriteLine(TabString(" Output file"))
            f.WriteLine(TabString("==========================================="))
            f.WriteLine(TabString("FilePath", Form_Main.txt_FilePath.Text))
            f.WriteLine(TabString("FileName", Form_Main.txt_FileName.Text))
            f.WriteLine(TabString("FileFormat", FileFormat))
            f.WriteLine(TabString("JpegQuality", Form_Main.txt_JpegQuality.NumericValueInteger.ToString))
            '
            f.WriteLine(TabString(""))
            f.WriteLine(TabString(" Tools"))
            f.WriteLine(TabString("==========================================="))
            'f.WriteLine(TabString("Tools_Run", Form_Main.Tools_Run.Checked.ToString))
            f.WriteLine(TabString("SlotRun", Form_Main.txt_SlotRun.Text))
            f.WriteLine(TabString("SlotStop", Form_Main.txt_SlotStop.Text))
            f.WriteLine(TabString("SlotWriteFile", Form_Main.txt_SlotWriteFile.Text))
            f.WriteLine(TabString("RisingSpeed", Form_Main.txt_RisingSpeed.NumericValueInteger.ToString))
            f.WriteLine(TabString("FallingSpeed", Form_Main.txt_FallingSpeed.NumericValueInteger.ToString))
            '
            f.WriteLine(TabString(""))
            f.WriteLine(TabString(" Video Input Params"))
            f.WriteLine(TabString("==========================================="))
            f.WriteLine(TabString("VideoInDevice", VideoInDevice))
            f.WriteLine(TabString("VideoFormat", VideoFormatParams.VideoFormat))
            f.WriteLine(TabString("VideoSize", VideoFormatParams.VideoSize))
            f.WriteLine(TabString("VideoFPS", VideoFormatParams.VideoFPS))
            '
            f.WriteLine(TabString(""))
            f.WriteLine(TabString(" Params"))
            f.WriteLine(TabString("==========================================="))
            f.WriteLine(TabString("Flip", Form_Main.chk_Flip.Checked.ToString))
            f.WriteLine(TabString("StartX", Form_Main.txt_StartX.NumericValueInteger.ToString))
            f.WriteLine(TabString("EndX", Form_Main.txt_EndX.NumericValueInteger.ToString))
            f.WriteLine(TabString("StartY", Form_Main.txt_StartY.NumericValueInteger.ToString))
            f.WriteLine(TabString("SizeY", Form_Main.txt_SizeY.NumericValueInteger.ToString))
            f.WriteLine(TabString("Dips", Form_Main.btn_Dips.Checked.ToString))
            f.WriteLine(TabString("Peaks", Form_Main.btn_Peaks.Checked.ToString))
            f.WriteLine(TabString("Colors", Form_Main.btn_Colors.Checked.ToString))
            f.WriteLine(TabString("Filter", Form_Main.txt_Filter.NumericValueInteger.ToString))
            f.WriteLine(TabString("MinNm", NanometersMin.ToString))
            f.WriteLine(TabString("MaxNm", NanometersMax.ToString))
            f.WriteLine(TabString("TrimPoint1", TrimPoint1.ToString))
            f.WriteLine(TabString("TrimPoint2", TrimPoint2.ToString))
            f.WriteLine(TabString("SpectrumFileSeparator", SpectrumFileSeparator))
        Catch
        End Try
        Try
            f.Close()
        Catch
        End Try
    End Sub


    Friend Sub Load_INI()
        ' ------------------------------------------------------------------------------- defaults
        VideoInDevice = ""
        FileFormat = "JPG"
        VideoFormatParams.VideoFormat = "RGB42"
        VideoFormatParams.VideoSize = "320 x 240"
        VideoFormatParams.VideoFPS = "30"
        Form_Main.Tools_Run.Checked = True
        ' -------------------------------------------------------------------------------
        ' ------------------------------------------------------------------------------- 
        ' With "Resume Next" subsequent parameters are loaded and f.Close() is executed
        ' -------------------------------------------------------------------------------
        On Error Resume Next  ' use Resume-Next instead of Try-Catch
        ' -------------------------------------------------------------------------------
        Dim l As String
        Dim iniFileName As String = Application.StartupPath & "\" & AssemblyName() & "_INI.txt"

        If My.Computer.FileSystem.FileExists(iniFileName) Then
            Dim f As System.IO.StreamReader
            f = IO.File.OpenText(iniFileName)
            Do While Not f.EndOfStream
                l = f.ReadLine()
                Select Case ExtractParamName(l)
                    Case "FormMain_Top" : Form_Main.Top = Val_Int(l)
                    Case "FormMain_Left" : Form_Main.Left = Val_Int(l)
                    Case "FormMain_Width" : Form_Main.Width = Val_Int(l)
                    Case "FormMain_Height" : Form_Main.Height = Val_Int(l)
                    Case "FormMain_WindowState" : Form_Main.WindowState = CType(Val(l), FormWindowState)
                        ' ------------------------------------------------------------------------------ 
                    Case "Form_VideoInControls_Top" : Form_VideoInControls.Top = Val_Int(l)
                    Case "Form_VideoInControls_Left" : Form_VideoInControls.Left = Val_Int(l)
                    Case "Form_VideoInControls_VisibleAtStart" : Form_VideoInControls_VisibleAtStart = l = "True"
                        ' ------------------------------------------------------------------------------
                        'Case "Tools_Run" : Form_Main.Tools_Run.Checked = l = "True"
                    Case "SlotRun" : Form_Main.txt_SlotRun.Text = l
                    Case "SlotStop" : Form_Main.txt_SlotStop.Text = l
                    Case "SlotWriteFile" : Form_Main.txt_SlotWriteFile.Text = l
                    Case "RisingSpeed" : Form_Main.txt_RisingSpeed.NumericValueInteger = Val_Int(l)
                    Case "FallingSpeed" : Form_Main.txt_FallingSpeed.NumericValueInteger = Val_Int(l)
                        ' ------------------------------------------------------------------------------
                    Case "VideoInDevice" : VideoInDevice = l
                    Case "VideoFormat" : VideoFormatParams.VideoFormat = l
                    Case "VideoSize" : VideoFormatParams.VideoSize = l
                    Case "VideoFPS" : VideoFormatParams.VideoFPS = l
                        ' ------------------------------------------------------------------------------
                    Case "Language" : Language = l
                        ' ------------------------------------------------------------------------------
                    Case "FilePath" : Form_Main.txt_FilePath.Text = l
                    Case "FileName" : Form_Main.txt_FileName.Text = l
                    Case "FileFormat" : FileFormat = l
                    Case "JpegQuality" : Form_Main.txt_JpegQuality.NumericValueInteger = Val_Int(l)
                        ' ------------------------------------------------------------------------------
                    Case "Flip" : Form_Main.chk_Flip.Checked = l = "True"
                    Case "StartX" : Form_Main.txt_StartX.NumericValueInteger = Val_Int(l)
                    Case "EndX" : Form_Main.txt_EndX.NumericValueInteger = Val_Int(l)
                    Case "StartY" : Form_Main.txt_StartY.NumericValueInteger = Val_Int(l)
                    Case "SizeY" : Form_Main.txt_SizeY.NumericValueInteger = Val_Int(l)
                    Case "Dips" : Form_Main.btn_Dips.Checked = l = "True"
                    Case "Peaks" : Form_Main.btn_Peaks.Checked = l = "True"
                    Case "Colors" : Form_Main.btn_Colors.Checked = l = "True"
                    Case "Filter" : Form_Main.txt_Filter.NumericValueInteger = Val_Int(l)
                    Case "MinNm" : NanometersMin = Val_Single(l)
                    Case "MaxNm" : NanometersMax = Val_Single(l)
                    Case "TrimPoint1" : TrimPoint1 = Val_Single(l)
                    Case "TrimPoint2" : TrimPoint2 = Val_Single(l)
                    Case "SpectrumFileSeparator" : SpectrumFileSeparator = l
                End Select
            Loop
            f.Close()
        End If
        LimitFormPosition(Form_Main)
        LimitFormPosition(Form_VideoInControls)
        '
    End Sub

    'Private Sub PrepareSaveLoadFolder()
    '    If My.Computer.FileSystem.FileExists(FileName) Then
    '        SaveLoadFolder = File_GetPath(FileName)
    '    End If
    '    If Not FolderExists(SaveLoadFolder) Then
    '        SaveLoadFolder = Application.StartupPath
    '    End If
    'End Sub

    Public Function File_GetPath(ByVal str As String) As String
        Try
            Return IO.Path.GetDirectoryName(str) & "\"
        Catch
            Return str
        End Try
    End Function


    ' ==================================================================================================
    '  SAVE IMAGE
    ' ==================================================================================================
    Public Sub SaveImage(ByVal img As Image, _
                         ByVal filename As String, _
                         ByVal extension As String, _
                         ByVal Quality As Int32)
        ' ---------------------------------------------------------------------
        extension = LCase(extension)
        filename = RemoveExtension(filename)
        filename += "." & extension
        ' ---------------------------------------------------------------------
        If img Is Nothing Then Exit Sub
        Try
            File_Kill(filename)
            If extension = "jpg" Then
                Dim ImageEncoders() As ImageCodecInfo = ImageCodecInfo.GetImageEncoders()
                Dim myEncoder As System.Drawing.Imaging.Encoder = System.Drawing.Imaging.Encoder.Quality
                Dim myEncoderParameters As New EncoderParameters(1)
                Dim myEncoderParameter As New EncoderParameter(myEncoder, Quality)
                myEncoderParameters.Param(0) = myEncoderParameter
                img.Save(filename, ImageEncoders(1), myEncoderParameters)
            Else
                img.Save(filename, ImageFormatFromFileExtension(extension))
            End If
        Catch
            MsgBox("Image save error", MsgBoxStyle.Exclamation)
        End Try
    End Sub

    Private Function ImageFormatFromFileExtension(ByVal extension As String) As ImageFormat
        Select Case LCase(extension)
            Case "jpg" : Return ImageFormat.Jpeg
            Case "png" : Return ImageFormat.Png
            Case "tiff" : Return ImageFormat.Tiff
            Case "exif" : Return ImageFormat.Exif
            Case "emf" : Return ImageFormat.Emf
            Case "wmf" : Return ImageFormat.Wmf
            Case "gif" : Return ImageFormat.Gif
            Case "bmp" : Return ImageFormat.Bmp
                'Case "ico" : Return ImageFormat.Icon
            Case Else : Return ImageFormat.Jpeg
        End Select
    End Function

    Friend Sub SaveImage(ByVal cnt As Control)
        If Form_Main.PBox_Spectrum.Image Is Nothing Then Exit Sub
        ' ---------------------------------------------------------------------
        If Not FolderExists(Form_Main.txt_FilePath.Text & "\") Then
            Form_Main.SelectSaveFolder()
        End If
        ' ---------------------------------------------------------------------
        Form_Main.txt_FilePath.Text = Trim(Form_Main.txt_FilePath.Text)
        Form_Main.txt_FileName.Text = Trim(Form_Main.txt_FileName.Text)
        ' ---------------------------------------------------------------------
        If Form_Main.txt_FileName.Text = "" Then
            Form_Main.txt_FileName.Text = "Noname_01"
        End If
        ' ---------------------------------------------------------------------
        Dim filename As String = Form_Main.txt_FilePath.Text & "\" & _
                                 Form_Main.txt_FileName.Text & "." & _
                                 Form_Main.ComboBox_FileType.Text
        ' --------------------------------------------------------------------- if exists increase index
        While My.Computer.FileSystem.FileExists(filename)
            filename = IncreaseFileNameIndex(filename)
            Form_Main.txt_FileName.Text = IO.Path.GetFileNameWithoutExtension(filename)
        End While
        ' --------------------------------------------------------------------- save
        SaveImage(GetImage(cnt), _
                  filename, _
                  Form_Main.ComboBox_FileType.Text, _
                  Form_Main.txt_JpegQuality.NumericValueInteger)
        ' --------------------------------------------------------------------- if exists increase index
        While My.Computer.FileSystem.FileExists(filename)
            filename = IncreaseFileNameIndex(filename)
            Form_Main.txt_FileName.Text = IO.Path.GetFileNameWithoutExtension(filename)
        End While
        ' ---------------------------------------------------------------------
        Beep()
    End Sub

    Private Function GetImage(ByVal cnt As Control) As Bitmap
        Dim s As Size = cnt.Size
        Dim bmp As Bitmap = New Bitmap(s.Width, s.Height)
        cnt.DrawToBitmap(bmp, New Rectangle(0, 0, s.Width, s.Height))
        Return bmp
    End Function


    ' ==================================================================================================
    '  SAVE SPECTRUM TO FILE
    ' ==================================================================================================
    Friend Sub SaveSpectrumToFile()
        If Not FolderExists(Form_Main.txt_FilePath.Text & "\") Then Return
        ' ---------------------------------------------------------------------
        Form_Main.txt_FilePath.Text = Trim(Form_Main.txt_FilePath.Text)

        'Dim fname As String = Filename_WithDateTime(Form_Main.txt_FilePath.Text, "SpectrumFile", ".txt")
        Dim fname As String = Filename_WithFirstFreeIndex(Form_Main.txt_FilePath.Text, "SpectrumFile", ".txt")

        Dim writer As System.IO.StreamWriter
        writer = My.Computer.FileSystem.OpenTextFileWriter(fname, _
                                                           False, _
                                                           New System.Text.UTF8Encoding(False))
        writer.Write(GetSpectrumText())
        writer.Close()
        Beep()
    End Sub

    'Friend Function Filename_WithDateTime(ByVal path As String, _
    '                                      ByVal name As String, _
    '                                      ByVal ext As String) As String

    '    Return path + "\" + name + "_" + Date.Now.ToString("yyyy-MM-dd_HH-mm-ss") + ext
    'End Function

    Friend Function Filename_WithFirstFreeIndex(ByVal path As String, _
                                                ByVal name As String, _
                                                ByVal ext As String) As String
        Dim index As Int32 = 1
        Dim completeFileName As String
        '
        Do
            completeFileName = path + "\" + name + "_" + index.ToString("000") + ext
            index += 1
        Loop Until Not IO.File.Exists(completeFileName)
        '
        Return completeFileName
    End Function

End Module
