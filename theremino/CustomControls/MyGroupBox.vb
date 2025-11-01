

' ================================================================================================================
'  MY GROUP BOX  -  Implements: "Border color" and "Border line type"
' ================================================================================================================

Imports System.ComponentModel
Imports System.Drawing
Imports System.Windows.Forms

Public Class MyGroupBox
    Inherits GroupBox

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







    Private Sub Initialize()
        _borderColor = Color.FromArgb(208, 208, 191)
        _borderStyle = ButtonBorderStyle.Solid
    End Sub


    Private _DimFactorGray As Integer = -35
    <Category("Appearance MyButton"), _
    Description("Get or Set how much to dim the grayed colors. Positive to Lighten and negative to Darken"), _
    DefaultValue(-35)> _
    Public Property DimFactorGray() As Integer
        Get
            Return _DimFactorGray
        End Get
        Set(ByVal Value As Integer)
            _DimFactorGray = Value
            Me.Invalidate()
        End Set
    End Property

    Private _borderColor As Color
    <Category("Appearance"), _
    Description("Get or Set the BorderColor")> _
    Public Property BorderColor() As Color
        Get
            Return _borderColor
        End Get
        Set(ByVal value As Color)
            If value <> _borderColor Then
                _borderColor = value
                DrawAll(Me.CreateGraphics)
            End If
        End Set
    End Property

    Private _borderStyle As ButtonBorderStyle
    <Category("Appearance"), _
    Description("Get or Set the BorderStyle")> _
    Public Property BorderStyle() As ButtonBorderStyle
        Get
            Return _borderStyle
        End Get
        Set(ByVal value As ButtonBorderStyle)
            If value <> _borderStyle Then
                _borderStyle = value
                DrawAll(Me.CreateGraphics)
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
                DrawAll(Me.CreateGraphics)
            End If
        End Set
    End Property


    Protected Overrides Sub OnPaint(ByVal e As PaintEventArgs)
        DrawAll(e.Graphics)
    End Sub

    Private Sub DrawAll(ByVal g As Graphics)
        If Not Me.Visible Then Exit Sub
        ' -----------------------------------------------------------------------
        Dim r As Rectangle
        Dim rf As RectangleF = g.VisibleClipBounds
        r.X = CInt(rf.X)
        r.Y = CInt(rf.Y)
        r.Width = CInt(rf.Width)
        r.Height = CInt(rf.Height)
        ' ---------------------------------------------------------------------- exclude spurious rectangles
        'Beep()
        'Debug.Print(r.ToString)
        If r.X <> 0 Or r.Y <> 0 Then Exit Sub
        ' ----------------------------------------------------------------------
        Dim tSize As Size = TextRenderer.MeasureText(Me.Text, Me.Font)
        Dim borderRect As Rectangle = r
        borderRect.Y = CInt(Math.Floor(borderRect.Y + (tSize.Height / 2)))
        borderRect.Height = CInt(Math.Floor(borderRect.Height - (tSize.Height / 2)))
        ControlPaint.DrawBorder(g, borderRect, Me._borderColor, _borderStyle)
        ' ----------------------------------------------------------------------
        Dim textRect As Rectangle = r
        textRect.X = (textRect.X + 12)
        textRect.Width = tSize.Width
        textRect.Height = tSize.Height

        ' ---------------------------------------------------------------------- draw the back rectangle
        If Me.BackColor.A = 0 And Me.Parent IsNot Nothing Then
            g.FillRectangle(New SolidBrush(Me.Parent.BackColor), textRect)
        Else
            g.FillRectangle(New SolidBrush(Me.BackColor), textRect)
        End If

        textRect.Width += 12
        ' ---------------------------------------------------------------------- draw the shadow
        If ShadowColor.A <> 0 Then
            textRect.Offset(1, 1)
            g.DrawString(Me.Text, Me.Font, _
                                  New SolidBrush(Me._ShadowColor), textRect)
            textRect.Offset(-1, -1)
        End If

        ' ---------------------------------------------------------------------- draw the text
        If Me.Enabled Then
            g.DrawString(Me.Text, Me.Font, New SolidBrush(Me.ForeColor), textRect)
        Else
            g.DrawString(Me.Text, Me.Font, New SolidBrush(GrayTheColor(Me.ForeColor)), textRect)
        End If

    End Sub


    Private Function GrayTheColor(ByVal GrayColor As Color) As Color
        Dim gray As Integer = CInt(GrayColor.R * 0.3 + GrayColor.G * 0.59 + GrayColor.B * 0.11)
        gray = CInt(gray + _DimFactorGray)
        If gray < 0 Then gray = 0
        If gray > 255 Then gray = 255
        Return Color.FromArgb(GrayColor.A, gray, gray, gray)
    End Function

End Class
