
Imports System.Drawing.Drawing2D
Imports System.Management

Module Module_Utils

    Friend Function CreateFillBrush(ByVal pbox As PictureBox, ByVal angle As Single, ByVal blend As Int32) As LinearGradientBrush   '创建线性渐变画笔;blend混合
        If pbox.ClientRectangle.Width < 1 Or pbox.ClientRectangle.Height < 1 Then
            Return Nothing
        End If
        CreateFillBrush = New LinearGradientBrush(pbox.ClientRectangle, Color.Black, Color.Black, angle)
        Dim myBlend As ColorBlend = New ColorBlend()
        Select Case blend
            Case 1 ' multicolor
                myBlend.Positions = New Single() {0.0F, 0.4F, 0.5F, 0.6F, 1.0F}
                myBlend.Colors = New Color() {Color.FromArgb(255, 0, 0),
                                              Color.FromArgb(255, 230, 0),
                                              Color.FromArgb(250, 250, 0),
                                              Color.FromArgb(230, 230, 0),
                                              Color.FromArgb(0, 100, 0)}

            Case 2 ' multicolor reverse
                myBlend.Positions = New Single() {0.0F, 0.4F, 0.5F, 0.6F, 1.0F}
                myBlend.Colors = New Color() {Color.FromArgb(0, 200, 0),
                                              Color.FromArgb(230, 230, 0),
                                              Color.FromArgb(250, 250, 0),
                                              Color.FromArgb(255, 230, 0),
                                              Color.FromArgb(255, 0, 0)}
            Case 3 ' green
                myBlend.Positions = New Single() {0.0F, 0.4F, 0.5F, 0.9F, 1.0F}
                myBlend.Colors = New Color() {Color.FromArgb(200, 250, 150),
                                              Color.FromArgb(80, 150, 100),
                                              Color.FromArgb(80, 150, 0),
                                              Color.FromArgb(80, 150, 0),
                                              Color.FromArgb(250, 0, 0)}
            Case 4 ' red
                myBlend.Positions = New Single() {0.0F, 0.4F, 0.5F, 0.9F, 1.0F}
                myBlend.Colors = New Color() {Color.FromArgb(255, 200, 160),
                                              Color.FromArgb(255, 150, 0),
                                              Color.FromArgb(250, 120, 0),
                                              Color.FromArgb(230, 80, 0),
                                              Color.FromArgb(60, 0, 0)}
            Case 5 ' blue
                myBlend.Positions = New Single() {0.0F, 0.4F, 0.5F, 0.9F, 1.0F}
                myBlend.Colors = New Color() {Color.FromArgb(110, 190, 220),
                                              Color.FromArgb(60, 130, 180),
                                              Color.FromArgb(0, 120, 160),
                                              Color.FromArgb(0, 100, 140),
                                              Color.FromArgb(0, 20, 80)}
        End Select
        CreateFillBrush.InterpolationColors = myBlend
    End Function


    ' =======================================================================================
    '  Utils
    ' =======================================================================================
    Friend Sub SleepMyThread(ByVal TimeMillisec As Int32)
        System.Threading.Thread.Sleep(TimeMillisec)
    End Sub
    Friend Function Shell_NormalFocus(ByVal path As String) As Int32
        Try
            Return Shell(path, AppWinStyle.NormalFocus)
        Catch
            MsgBox("Cannot open the path: " & vbCr & path, MsgBoxStyle.Information)
            Return 0
        End Try
    End Function
    Friend Function MouseButtonLeftPressed() As Boolean
        Return (Control.MouseButtons And Windows.Forms.MouseButtons.Left) <> Windows.Forms.MouseButtons.None
    End Function
    Friend Function MouseButtonRightPressed() As Boolean
        Return (Control.MouseButtons And Windows.Forms.MouseButtons.Right) <> Windows.Forms.MouseButtons.None
    End Function

    Friend Sub Swap(ByRef n1 As Single, ByRef n2 As Single)
        Dim n3 As Single = n2
        n2 = n1
        n1 = n3
    End Sub
    Friend Sub Swap(ByRef n1 As Double, ByRef n2 As Double)
        Dim n3 As Double = n2
        n2 = n1
        n1 = n3
    End Sub
    Friend Sub Swap(ByRef n1 As Int32, ByRef n2 As Int32)
        Dim n3 As Int32 = n2
        n2 = n1
        n1 = n3
    End Sub


    ' ==============================================================================================================
    '   COMBO FUNCTIONS
    ' ==============================================================================================================
    Friend Sub Combo_Init(ByVal combo As ComboBox, ByVal str As String)
        Dim old As Boolean = EventsAreEnabled
        EventsAreEnabled = False
        If str = Nothing Then str = ""
        With combo
            .Items.Clear()
            .Items.Add(str)
            .SelectedIndex = 0
        End With
        EventsAreEnabled = old
    End Sub
    Friend Sub Combo_SetIndex_FromString(ByVal combo As ComboBox, ByVal str As String)
        Dim old As Boolean = EventsAreEnabled
        EventsAreEnabled = False
        If str = Nothing Then str = ""
        With combo
            For i As Int32 = 0 To .Items.Count - 1
                If .Items(i).ToString = str Then
                    .SelectedIndex = i
                    Exit For
                End If
            Next
        End With
        EventsAreEnabled = old
    End Sub
    Friend Function Combo_GetValue(ByVal combo As ComboBox) As String
        If combo.SelectedIndex < 0 Then Return ""
        Return combo.Items(combo.SelectedIndex).ToString()
    End Function
    Friend Sub Combo_SetIndex(ByVal combo As ComboBox, ByVal index As Int32)
        If combo.Items.Count < 1 Then Exit Sub
        If index < 0 Then index = 0
        If index > combo.Items.Count - 1 Then index = combo.Items.Count - 1
        combo.SelectedIndex = index
    End Sub


    ' =============================================================================================
    '  ASYNC KEY and MOUSE STATE
    ' =============================================================================================
    Private Const PRESSED_NOW As Int32 = &H8000
    Private Const PRESSED_AFTER_PREVIOUS_CALL As Int32 = &H1
    Private Declare Function GetAsyncKeyState Lib "user32" (ByVal vKey As Int32) As Int16
    Friend Function KeyFromPreviousCall(ByVal k As Int32) As Boolean
        KeyFromPreviousCall = (GetAsyncKeyState(k) And PRESSED_AFTER_PREVIOUS_CALL) <> 0
    End Function
    Friend Function Key(ByVal k As Int32) As Boolean
        Key = (GetAsyncKeyState(k) And PRESSED_NOW) <> 0
    End Function
    'Friend Sub WaitMouseOff()
    '    While Key(Keys.LButton) Or Key(Keys.MButton) Or Key(Keys.RButton)
    '        SleepMyThread(1)
    '    End While
    'End Sub
    'Friend Sub WaitKeyOff(ByVal WaitKey As Long)
    '    Do
    '        SleepMyThread(1)
    '    Loop Until Not Key(WaitKey)
    'End Sub


    ' ================================================================================
    '  Wavelength To Color
    ' ================================================================================
    Friend Function WavelengthToColor(ByVal Wavelength As Single) As Color
        Dim Blue As Single
        Dim Green As Single
        Dim Red As Single
        Dim Factor As Single
        If Wavelength >= 380 AndAlso Wavelength < 440 Then
            Red = -1 * (Wavelength - 440) / (440 - 380)
            Green = 0
            Blue = 1
        ElseIf Wavelength >= 440 AndAlso Wavelength < 490 Then
            Red = 0
            Green = (Wavelength - 440) / (490 - 440)
            Blue = 1
        ElseIf Wavelength >= 490 AndAlso Wavelength < 510 Then
            Red = 0
            Green = 1
            Blue = -1 * (Wavelength - 510) / (510 - 490)
        ElseIf Wavelength >= 510 AndAlso Wavelength < 580 Then
            Red = (Wavelength - 510) / (580 - 510)
            Green = 1
            Blue = 0
        ElseIf Wavelength >= 580 AndAlso Wavelength < 645 Then
            Red = 1
            Green = -1 * (Wavelength - 645) / (645 - 580)
            Blue = 0
        ElseIf Wavelength >= 645 AndAlso Wavelength <= 780 Then
            Red = 1
            Green = 0
            Blue = 0
        Else
            Red = 0
            Green = 0
            Blue = 0
        End If
        ' ------------------------------------------------------------ gradual dim to invisible ranges
        Const IR2 As Single = 780 ' original = 780
        Const IR1 As Single = 650 ' original = 700
        Const UV2 As Single = 420
        Const UV1 As Single = 380
        If Wavelength > IR2 Or Wavelength < UV1 Then
            Factor = 0
        ElseIf Wavelength > IR1 Then
            Factor = (IR2 - Wavelength) / (IR2 - IR1)
        ElseIf (Wavelength < UV2) Then
            Factor = (Wavelength - UV1) / (UV2 - UV1)
        Else
            Factor = 1
        End If
        ' ------------------------------------------------------------ make the color
        Return Color.FromArgb(CInt(255 * Red * Factor), _
                              CInt(255 * Green * Factor), _
                              CInt(255 * Blue * Factor))
    End Function


    ' =======================================================================================================
    '   REDIM PRESERVE BIDIMENSIONAL STRING ARRAY
    ' =======================================================================================================
    Friend Sub RedimPreserve_2D_Array(ByRef ar(,) As String, ByVal rowUpperIdx As Int32, ByVal colUpperIdx As Int32)
        Dim newArray(rowUpperIdx, colUpperIdx) As String
        Dim minRows As Integer = Math.Min(rowUpperIdx + 1, ar.GetLength(0))
        Dim minCols As Integer = Math.Min(colUpperIdx + 1, ar.GetLength(1))
        For i As Int32 = 0 To minRows - 1
            For j As Int32 = 0 To minCols - 1
                newArray(i, j) = ar(i, j)
            Next
        Next
        ar = newArray
    End Sub


    ' =======================================================================================================
    '   FadeIn and FadeOut 
    ' =======================================================================================================
    Friend Sub Forms_FadeTo(ByVal FinalValue As Double, ByVal TimeMillisec As Double)
        Try
            If TimeMillisec < 1 Then TimeMillisec = 1
            Dim v As Double
            Dim k As Double
            Dim date1 As Date
            '
            Dim StartValue As Double = Form_Main.Opacity
            '
            Application.DoEvents()
            System.Threading.Thread.Sleep(1)
            date1 = Date.Now
            Do
                k = Date.Now.Subtract(date1).TotalMilliseconds / TimeMillisec
                If k > 1 Then k = 1
                v = StartValue + (FinalValue - StartValue) * k
                If FinalValue = 0 Then
                    v = v * 0.5
                End If
                Form_Main.Opacity = v
                Form_VideoInControls.Opacity = v
                System.Threading.Thread.Sleep(20)
                'Debug.Print(v.ToString)
            Loop Until k >= 1
        Catch
            Form_Main.Opacity = 1
            Form_VideoInControls.Opacity = 1
        End Try
    End Sub


    ' ===============================================================
    '  WINDOWS SCHEDULER PRECISION
    ' =============================================================== 
    '<System.Runtime.InteropServices.DllImport("winmm.dll")> _
    'Private Function timeBeginPeriod(ByVal uPeriod As Int32) As Int32
    'End Function
    '<System.Runtime.InteropServices.DllImport("winmm.dll")> _
    'Private Function timeEndPeriod(ByVal uPeriod As Int32) As Int32
    'End Function
    'Friend Sub MillisecondPrecision_Start()
    '    timeBeginPeriod(1)
    'End Sub
    'Friend Sub MillisecondPrecision_End()
    '    timeEndPeriod(1)
    'End Sub


    'Public Function VarPtr(ByVal e As Object) As IntPtr
    '    Dim GC As System.Runtime.InteropServices.GCHandle = _
    '    System.Runtime.InteropServices.GCHandle.Alloc(e, System.Runtime.InteropServices.GCHandleType.Pinned)
    '    Dim GC2 As Int32 = GC.AddrOfPinnedObject.ToInt32
    '    GC.Free()
    '    Return CType(GC2, IntPtr)
    'End Function

    'Public Function VarPtr_Int(ByVal e As Object) As Int32
    '    Dim GC As System.Runtime.InteropServices.GCHandle = _
    '    System.Runtime.InteropServices.GCHandle.Alloc(e, System.Runtime.InteropServices.GCHandleType.Pinned)
    '    Dim GC2 As Int32 = GC.AddrOfPinnedObject.ToInt32
    '    GC.Free()
    '    Return GC2
    'End Function

End Module
