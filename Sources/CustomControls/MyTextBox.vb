
' ================================================================================================================
'  MY TEXT BOX  -  Implements: "Mouse edit", "BorderColor", "BorderStyle" and "BackColor_Over"
' ----------------------------------------------------------------------------------------------------------------
'  Version with "MyTextBox_MouseMove" corrected the 2016/12/11 for tablet errors
'  
' ================================================================================================================

Imports System.ComponentModel
Imports System.Drawing
Imports System.Windows.Forms


Public Class MyTextBox
    Inherits TextBox

    ' =================================================================================== 
    '      required by Designer   -    do not modify with the code editor
    ' ===================================================================================
    Public Sub New()
        MyBase.New()
        InitializeComponent()
        ' -------------------------- my inizializations
        Initialize()
    End Sub

    <System.Diagnostics.DebuggerNonUserCode()> _
    Protected Overrides Sub Dispose(ByVal disposing As Boolean)
        If disposing Then
            If (Not (components) Is Nothing) Then
                components.Dispose()
            End If
        End If
        MyBase.Dispose(disposing)
    End Sub

    Private components As System.ComponentModel.Container = Nothing

    '<System.Diagnostics.DebuggerStepThrough()> _
    Private Sub InitializeComponent()
        components = New System.ComponentModel.Container
    End Sub
    ' ===================================================================================


    Private Declare Function GetCaretPos Lib "user32" (ByRef lpPoint As Point) As Integer
    Private Declare Function HideCaret Lib "user32" (ByVal hWnd As IntPtr) As Boolean
    Private Declare Function ShowCaret Lib "user32" (ByVal hWnd As IntPtr) As Boolean


    Private Sub Initialize()
        'SetStyle(ControlStyles.AllPaintingInWmPaint Or _
        '        ControlStyles.DoubleBuffer Or _
        '        ControlStyles.ResizeRedraw Or _
        '        ControlStyles.SupportsTransparentBackColor Or _
        '        ControlStyles.UserPaint, True)

        SetStyle(ControlStyles.AllPaintingInWmPaint Or _
                 ControlStyles.DoubleBuffer Or _
                 ControlStyles.ResizeRedraw Or _
                 ControlStyles.SupportsTransparentBackColor, True)
    End Sub


#Region "Properties"

    Private _DimFactorGray As Integer = -35
    <Category("Appearance MyButton"), _
   Description("Get or Set how much to dim the grayed colors. Positive to Lighten and negative to Darken"), _
   DefaultValue(-35)> _
    Public Property DimFactorGray() As Integer
        Get
            Return _DimFactorGray
        End Get
        Set(ByVal Value As Integer)
            If Value <> _DimFactorGray Then
                _DimFactorGray = Value
                Me.Invalidate()
            End If
        End Set
    End Property

    Private _BackColor_Over As Color = Me.BackColor
    <Category("Appearance"), _
    Description("Get or Set the BackColor when the mouse is Over the TextBox")> _
    Public Property BackColor_Over() As Color
        Get
            Return _BackColor_Over
        End Get
        Set(ByVal Value As Color)
            If Value <> _BackColor_Over Then
                _BackColor_Over = Value
                Me.Invalidate()
            End If
        End Set
    End Property

    Private _ShadowColor As Color = Color.Transparent
    <Category("Appearance"), _
    Description("Get or Set the TextShadowColor")> _
    Public Property ShadowColor() As Color
        Get
            Return _ShadowColor
        End Get
        Set(ByVal Value As Color)
            If Value <> _ShadowColor Then
                _ShadowColor = Value
                Me.Invalidate()
            End If
        End Set
    End Property

    Private _BorderColor As Color = Color.Transparent
    <Category("Appearance"), _
    Description("Get or Set the BorderColor")> _
    Public Property RectangleColor() As Color
        Get
            Return _BorderColor
        End Get
        Set(ByVal value As Color)
            If value <> _BorderColor Then
                _BorderColor = value
                Me.Invalidate()
            End If
        End Set
    End Property

    Private _BorderStyle As ButtonBorderStyle
    <Category("Appearance"), _
    Description("Get or Set the BorderStyle")> _
    Public Property RectangleStyle() As ButtonBorderStyle
        Get
            Return _BorderStyle
        End Get
        Set(ByVal value As ButtonBorderStyle)
            If value <> _BorderStyle Then
                _BorderStyle = value
                Me.Invalidate()
            End If
        End Set
    End Property

    Private _MinValue As Double = 0
    <Category("Mouse and Arrows edit"), _
    Description("Get or Set the minimum value"), _
    DefaultValue(0)> _
    Public Property MinValue() As Double
        Get
            Return _MinValue
        End Get
        Set(ByVal value As Double)
            If value <> _MinValue Then
                _MinValue = value
            End If
        End Set
    End Property

    Private _MaxValue As Double = 100
    <Category("Mouse and Arrows edit"), _
    Description("Get or Set the maximum value"), _
    DefaultValue(0)> _
    Public Property MaxValue() As Double
        Get
            Return _MaxValue
        End Get
        Set(ByVal value As Double)
            If value <> _MaxValue Then
                _MaxValue = value
            End If
        End Set
    End Property

    Private _Increment As Double = 0
    <Category("Mouse and Arrows edit"), _
    Description("Get or Set the increment when editing with mouse ( use Increment=0 to disable Mouse edit "), _
    DefaultValue(0)> _
    Public Property Increment() As Double
        Get
            Return _Increment
        End Get
        Set(ByVal value As Double)
            If value < 0 Then value = 0
            If value <> _Increment Then
                _Increment = value
            End If
        End Set
    End Property

    Private _ArrowsIncrement As Double = 0
    <Category("Mouse and Arrows edit"), _
    Description("Get or Set the increment when editing with the UP-DOWN arrows ( use ArrowsIncrement=0 to disable Arrows-edit "), _
    DefaultValue(0)> _
    Public Property ArrowsIncrement() As Double
        Get
            Return _ArrowsIncrement
        End Get
        Set(ByVal value As Double)
            If value < 0 Then value = 0
            If value <> _ArrowsIncrement Then
                _ArrowsIncrement = value
            End If
        End Set
    End Property

    Private _Decimals As Int32 = 0
    <Category("Mouse and Arrows edit"), _
    Description("Get or Set the number of decimals"), _
    DefaultValue(0)> _
    Public Property Decimals() As Int32
        Get
            Return _Decimals
        End Get
        Set(ByVal value As Int32)
            If value < 0 Then value = 0
            If value > 16 Then value = 16
            If value <> _Decimals Then
                _Decimals = value
            End If
        End Set
    End Property

    Private _RoundingStep As Double = 0
    <Category("Mouse and Arrows edit"), _
    Description("Get or Set the rounding step when editing values. ( use RoundingStep=0 to disable rounding ) "), _
    DefaultValue(0)> _
    Public Property RoundingStep() As Double
        Get
            Return _RoundingStep
        End Get
        Set(ByVal value As Double)
            _RoundingStep = value
        End Set
    End Property

    <Category("Mouse and Arrows edit"), _
    Description("Get or Set the numeric value (double)"), _
    DefaultValue(0)> _
    Public Property NumericValue() As Double
        Get
            Dim n As Double = Val(Replace(Me.Text, ",", "."))
            If n > _MaxValue Then n = _MaxValue
            If n < _MinValue Then n = _MinValue
            Return n
        End Get
        Set(ByVal value As Double)
            If value > _MaxValue Then value = _MaxValue
            If value < _MinValue Then value = _MinValue
            If RoundingStep > 0 Then
                value = CInt(value / RoundingStep) * RoundingStep
            End If
            Dim s As String
            If _SuppressZeros Then
                s = Replace(Format(value, "0." & StrDup(_Decimals, "#")), ",", ".")
            Else
                s = Replace(Format(value, "0." & StrDup(_Decimals, "0")), ",", ".")
            End If
            If s <> Me.Text Then
                Me.Text = s
                Me.Invalidate()
            End If
        End Set
    End Property

    <Category("Mouse and Arrows edit"), Description("Get or Set the numeric value (integer)"), DefaultValue(0)>
    Public Property NumericValueInteger() As Int32
        Get
            Dim n As Double = Val(Me.Text)
            If n > _MaxValue Then n = CInt(_MaxValue)
            If n < _MinValue Then n = CInt(_MinValue)
            Return CInt(n)
        End Get
        Set(ByVal value As Int32)
            If value > _MaxValue Then value = CInt(_MaxValue)
            If value < _MinValue Then value = CInt(_MinValue)
            If RoundingStep > 0 Then
                value = CInt(CInt(value / RoundingStep) * RoundingStep)
            End If
            Dim s As String = value.ToString
            If s <> Me.Text Then
                Me.Text = s
                Me.Invalidate()
            End If
        End Set
    End Property

    Private _SuppressZeros As Boolean
    <Category("Mouse and Arrows edit"), _
    Description("True to suppress trailing zeros"), _
    DefaultValue(False)> _
    Public Property SuppressZeros() As Boolean
        Get
            Return _SuppressZeros
        End Get
        Set(ByVal value As Boolean)
            If _SuppressZeros <> value Then
                _SuppressZeros = value
            End If
        End Set
    End Property



#End Region 'Properties


#Region "Events"

    Private Enum eDrawState
        Over
        Normal
    End Enum
    Private _DrawState As eDrawState = eDrawState.Normal

    Private _TrimmingValue As Boolean = False
    Private _EditingPos As Point
    Private _StartEditPos As Point
    Private _DeltaY As Int32
    Private _StartValue As Double


    Private Sub MyTextBox_MouseDown(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles Me.MouseDown
        _EditingPos = Cursor.Position
        _StartEditPos = Cursor.Position
    End Sub

    Private Sub MyTextBox_MouseMove(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles Me.MouseMove

        ' CursorPosOldY is used for tablet errors (continuous increment also whith mouse not moving)
        Static CursorPositionOld As Point

        If Not _TrimmingValue And Me.Enabled And _Increment > 0 And e.Button = Windows.Forms.MouseButtons.Left Then
            If Math.Abs(Cursor.Position.Y - _EditingPos.Y) > Math.Abs(Cursor.Position.X - _EditingPos.X) + 1 Then
                _DeltaY = 0
                HideCaret_WithoutTestIt()
                _StartValue = Me.NumericValue
                LimitValue_Double(_StartValue, _MinValue, _MaxValue)
                _TrimmingValue = True
                HideCursor()
                CursorPositionOld = Cursor.Position
            End If
        End If

        If _TrimmingValue Then
            If Cursor.Position <> CursorPositionOld Then
                CursorPositionOld = Cursor.Position
                ' ---------------------------------------------------- fixed speed
                '_DeltaY += (Cursor.Position.Y - _EditingPos.Y)
                ' ---------------------------------------------------- variable speed
                Dim dy As Int32 = Cursor.Position.Y - _EditingPos.Y
                Dim absdy As Double = Math.Abs(dy)
                Dim signdy As Int32 = Math.Sign(dy)
                If absdy > 150 Then absdy = 150
                _DeltaY += CInt(signdy * absdy ^ 1.4)
                ' ---------------------------------------------------- calc the new value
                Dim v As Double = _StartValue - _DeltaY * _Increment
                ' ---------------------------------------------------- limit the value
                If LimitValue_Double(v, _MinValue, _MaxValue) Then
                    _DeltaY = CInt((_StartValue - v) / _Increment)
                End If
                ' ---------------------------------------------------- set the value and draw it
                Me.NumericValue = v
                Me.Invalidate()
                'Me.DrawAll(Me.CreateGraphics())
                ' ---------------------------------------------------- cursor position to the right of the textbox 
                Me.SelectionStart = 99
                _EditingPos.X = Me.PointToScreen(New Point(Me.Width, 0)).X
                Cursor.Position = _EditingPos
                ' ---------------------------------------------------- wait
                System.Threading.Thread.Sleep(10)
            End If
        End If
    End Sub

    Protected Overrides Sub OnMouseEnter(ByVal e As System.EventArgs)
        MyBase.OnMouseEnter(e)
        If SelectionLength = 0 Then
            _DrawState = eDrawState.Over
            Me.Invalidate()
        End If
    End Sub

    Private Sub MyTextBox_MouseLeave(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.MouseLeave
        If _TrimmingValue Then
            _TrimmingValue = False
            ShowCaret_WithoutTestIt()
            ShowCursor()
            Me.SelectionStart = 999
        End If
        '
        If SelectionLength = 0 Then
            _DrawState = eDrawState.Normal
            Me.Invalidate()
        End If
    End Sub

    Private Sub MyTextBox_LostFocus(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.LostFocus
        SelectionLength = 0
        _DrawState = eDrawState.Normal
        If _Increment > 0 Or _ArrowsIncrement > 0 Then
            MyTextBox_LimitValue()
        End If
        Me.Invalidate()
    End Sub

    Private Sub MyTextBox_LimitValue()
        Dim v As Double = Val(Me.Text)
        If v > _MaxValue Then v = _MaxValue
        If v < _MinValue Then v = _MinValue
        If RoundingStep > 0 Then
            v = CInt(v / RoundingStep) * RoundingStep
        End If
        Dim s As String
        If _SuppressZeros Then
            s = Replace(Format(v, "0." & StrDup(_Decimals, "#")), ",", ".")
        Else
            s = Replace(Format(v, "0." & StrDup(_Decimals, "0")), ",", ".")
        End If
        If s <> Me.Text Then
            Me.Clear()
            Me.Text = s
            Me.Invalidate()
        End If
    End Sub


    Private Sub MyTextBox_MouseUp(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles Me.MouseUp
        If _TrimmingValue Then
            _TrimmingValue = False
            Cursor.Position = _StartEditPos
            ShowCaret_WithoutTestIt()
            ShowCursor()
            Me.SelectionStart = 999
        End If
    End Sub

    Private Function LimitValue_Double(ByRef v As Double, ByVal min As Double, ByVal max As Double) As Boolean
        If v > max Then
            v = max
            Return True
        End If
        If v < min Then
            v = min
            Return True
        End If
        Return False
    End Function

    Private Sub MyTextBox_KeyDown(ByVal sender As Object, ByVal e As System.Windows.Forms.KeyEventArgs) Handles Me.KeyDown
        If Me.Enabled And _ArrowsIncrement > 0 Then
            Dim n As Single = 0
            Select Case e.KeyCode
                ' ------------------------------------------------- Up and Down = *1
                Case Keys.Up : n = 1
                Case Keys.Down : n = -1
                    ' --------------------------------------------- PageUp and PageDown = *10
                Case Keys.PageUp : n = 10
                Case Keys.PageDown : n = -10
            End Select
            If n <> 0 Then
                ' ------------------------------------------------- SHIFT, CTRL and ALT modifiers
                If e.Shift Then n *= 100
                If e.Control Then n *= 10
                If e.Alt Then n /= 10
                ' ------------------------------------------------- 
                Me.NumericValue += _ArrowsIncrement * n
                HideCaret()
            End If
        End If
    End Sub

    Private Sub MyTextBox_KeyUp(ByVal sender As Object, ByVal e As System.Windows.Forms.KeyEventArgs) Handles Me.KeyUp
        If e.KeyValue = Keys.Up Or e.KeyValue = Keys.Down Then
            Me.SelectionStart = 999
        End If
        ShowCaret()
    End Sub

    Private Sub MyTextBox_MouseWheel(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles Me.MouseWheel
        If Me.Enabled And _ArrowsIncrement > 0 Then
            ' -------------------------------------- this method works also with micrometric mouses like Microsoft Mouse
            Dim n As Double = e.Delta / 120.0F
            If n > 0 And n < 1 Then n = 1
            If n < 0 And n > -1 Then n = -1
            ' ------------------------------------------------- SHIFT, CTRL and ALT modifiers
            If My.Computer.Keyboard.ShiftKeyDown Then n *= 100
            If My.Computer.Keyboard.CtrlKeyDown Then n *= 10
            If My.Computer.Keyboard.AltKeyDown Then n /= 10
            ' -------------------------------------------------
            Me.NumericValue += _ArrowsIncrement * n
            ' ------------------------------------------------- 
            'HideCaret()
            Me.SelectionStart = 999
        End If
    End Sub



    ' ------------------------------------------------------------------------------------------
    '  CARET CONTROL ( HIDE and SHOW only one time )
    ' ------------------------------------------------------------------------------------------
    Private CaretIsVisible As Boolean = True
    Private Sub HideCaret()
        If CaretIsVisible Then
            If OperatingSystemIsWindows Then HideCaret(Me.Handle)
            CaretIsVisible = False
        End If
    End Sub
    Private Sub ShowCaret()
        If Not CaretIsVisible Then
            If OperatingSystemIsWindows Then ShowCaret(Me.Handle)
            CaretIsVisible = True
        End If
    End Sub

    Private Sub HideCaret_WithoutTestIt()
        If OperatingSystemIsWindows Then HideCaret(Me.Handle)
    End Sub
    Private Sub ShowCaret_WithoutTestIt()
        If OperatingSystemIsWindows Then ShowCaret(Me.Handle)
    End Sub


    ' ------------------------------------------------------------------------------------------
    '  CURSOR CONTROL ( HIDE and SHOW )
    ' ------------------------------------------------------------------------------------------
    'Private BlankCursor As Cursor = New Cursor(New IO.MemoryStream(My.Resources.Blank))
    Private Sub HideCursor()
        Cursor.Hide()
        'Me.Cursor = BlankCursor
    End Sub
    Private Sub ShowCursor()
        Cursor.Show()
        'Me.Cursor = Cursors.IBeam
    End Sub


#End Region 'Events


#Region "Drawing"


    <System.Diagnostics.DebuggerStepThrough()> _
    Protected Overrides Sub WndProc(ByRef m As System.Windows.Forms.Message)
        MyBase.WndProc(m)
        '
        Const WM_PAINT As Int32 = 15
        If m.Msg = WM_PAINT Then
            Me.DrawAll(Me.CreateGraphics())
        End If
    End Sub

    'Protected Overrides Sub OnPaint(ByVal e As System.Windows.Forms.PaintEventArgs)
    '    'MyBase.OnPaint(e)

    '    DrawAll(e.Graphics)
    'End Sub

    'Private Declare Function LockWindowUpdate Lib "user32" (ByVal hWnd As Integer) As Integer


    Private Sub DrawAll(ByVal g As Graphics)

        If Me.Visible Then

            'LockWindowUpdate(Me.Handle.ToInt32)


            Dim r As Rectangle

            If Me.RectangleStyle <> ButtonBorderStyle.None And Me.RectangleColor.A <> 0 Then
                r = New Rectangle(0, 0, Me.Width, Me.Height)
                ControlPaint.DrawBorder(g, r, Me.RectangleColor, Me.RectangleStyle)
                '
                'Debug.Print(r.ToString)
                'Beep()
            End If


            HideCaret_WithoutTestIt()
            ' ------------------------------------------------------------------ background of the text area
            If BackColor_Over.A <> 0 Or Not Me.Enabled Then
                r = Me.ClientRectangle
                r.Inflate(-1, -1)
                If Me.Enabled Then
                    If _DrawState = eDrawState.Over Then
                        g.FillRectangle(New SolidBrush(BackColor_Over), r)
                    Else
                        g.FillRectangle(New SolidBrush(BackColor), r)
                    End If
                Else
                    g.FillRectangle(New SolidBrush(GrayTheColor(BackColor)), r)
                End If
            End If

            ' ------------------------------------------------------------------ draw the text
            r = Me.ClientRectangle
            'r.X -= 1
            'r.Y -= 1
            'r.Width += 2
            'r.Height += 2
            ' ----------------------------------------------------------------- draw the Shadow
            If ShadowColor.A <> 0 Then
                r.Offset(1, 1)
                g.DrawString(Me.Text, Me.Font, _
                                      New SolidBrush(Me._ShadowColor), r, GetStringFormat(Me.TextAlign))
                r.Offset(-1, -1)
            End If
            ' ------------------------------------------------------------------ draw the text
            'g.TextRenderingHint = Drawing.Text.TextRenderingHint.AntiAliasGridFit
            g.DrawString(Me.Text, Me.Font, New SolidBrush(Me.ForeColor), r, GetStringFormat(Me.TextAlign))
            '
            ShowCaret_WithoutTestIt()

            g.Dispose()

            'LockWindowUpdate(0)

        End If
    End Sub

    Private Function GetStringFormat(ByVal ctrlalign As HorizontalAlignment) As StringFormat
        Dim strFormat As StringFormat = New StringFormat()
        Select Case ctrlalign
            Case HorizontalAlignment.Center
                strFormat.Alignment = StringAlignment.Center
            Case HorizontalAlignment.Left
                strFormat.Alignment = StringAlignment.Near
            Case HorizontalAlignment.Right
                strFormat.Alignment = StringAlignment.Far
        End Select
        '
        If Me.BorderStyle = Windows.Forms.BorderStyle.None Or Me.Multiline Then
            strFormat.LineAlignment = StringAlignment.Near
        Else
            strFormat.LineAlignment = StringAlignment.Center
        End If
        '
        Return strFormat
    End Function

    Private Function GrayTheColor(ByVal GrayColor As Color) As Color
        Dim gray As Integer = CInt(GrayColor.R * 0.3 + GrayColor.G * 0.59 + GrayColor.B * 0.11)
        gray = CInt(gray + _DimFactorGray)
        If gray < 0 Then gray = 0
        If gray > 255 Then gray = 255
        Return Color.FromArgb(GrayColor.A, gray, gray, gray)
    End Function


#End Region 'Drawing



End Class



