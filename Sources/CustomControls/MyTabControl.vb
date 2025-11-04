

' ================================================================================================================
'  MY TAB CONTROL   -  Implements: "Header colors"   BackColor_Selected1 / BackColor_Selected2 etc...
' ================================================================================================================

Imports System.ComponentModel
Imports System.Windows.Forms
Imports System.Drawing

Public Class MyTabControl
    Inherits System.Windows.Forms.TabControl


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
        '
        SetStyle(ControlStyles.AllPaintingInWmPaint Or ControlStyles.DoubleBuffer Or _
                                                        ControlStyles.ResizeRedraw Or _
                                                        ControlStyles.UserPaint, _
                                                        True)
        '
        _BackColor_Selected1 = Color.FromArgb(80, 60, 180)
        _BackColor_Selected2 = Color.FromArgb(30, 0, 50)
        _BackColor_Unselected = Color.FromArgb(0, 0, 0)
        _ForeColor_Selected = Color.FromArgb(170, 255, 0)
        _ForeColor_Unselected = Color.FromArgb(160, 160, 160)
    End Sub


    ' --------------------------------------------------------------------------------------------------
    '  This helps to restore "UseVisualStyleBackColor = False" after each "TabPages Collection" editing 
    ' --------------------------------------------------------------------------------------------------
    Private Sub MyTabControl_ControlAdded(ByVal sender As Object, ByVal e As System.Windows.Forms.ControlEventArgs) Handles Me.ControlAdded
        For Each tp As TabPage In TabPages
            tp.UseVisualStyleBackColor = False
        Next
    End Sub




    Private _BackColor_Selected1 As Color
    <Category("Appearance"), _
    Description("Get or Set the first Color for selected tabs")> _
    Public Property BackColor_Selected1() As Color
        Get
            Return _BackColor_Selected1
        End Get
        Set(ByVal value As Color)
            If value <> _BackColor_Selected1 Then
                _BackColor_Selected1 = value
                _Paint(Me.CreateGraphics, False)
            End If
        End Set
    End Property

    Private _BackColor_Selected2 As Color
    <Category("Appearance"), _
    Description("Get or Set the second Color for selected tabs")> _
    Public Property BackColor_Selected2() As Color
        Get
            Return _BackColor_Selected2
        End Get
        Set(ByVal value As Color)
            If value <> _BackColor_Selected2 Then
                _BackColor_Selected2 = value
                _Paint(Me.CreateGraphics, False)
            End If
        End Set
    End Property

    Private _BackColor_Unselected As Color
    <Category("Appearance"), _
    Description("Get or Set the BackColor for unselected tabs")> _
    Public Property BackColor_UnSelected() As Color
        Get
            Return _BackColor_Unselected
        End Get
        Set(ByVal value As Color)
            If value <> _BackColor_Unselected Then
                _BackColor_Unselected = value
                _Paint(Me.CreateGraphics, True)
            End If
        End Set
    End Property

    Private _ForeColor_Selected As Color
    <Category("Appearance"), _
    Description("Get or Set the ForeColor for selected tabs")> _
    Public Property ForeColor_Selected() As Color
        Get
            Return _ForeColor_Selected
        End Get
        Set(ByVal value As Color)
            If value <> _ForeColor_Selected Then
                _ForeColor_Selected = value
                _Paint(Me.CreateGraphics, False)
            End If
        End Set
    End Property

    Private _ForeColor_Unselected As Color
    <Category("Appearance"), _
    Description("Get or Set the ForeColor for unselected tabs")> _
    Public Property ForeColor_Unselected() As Color
        Get
            Return _ForeColor_Unselected
        End Get
        Set(ByVal value As Color)
            If value <> _ForeColor_Unselected Then
                _ForeColor_Unselected = value
                _Paint(Me.CreateGraphics, False)
            End If
        End Set
    End Property

    Private _ShadowColor_Selected As Color = Color.Transparent
    <Category("Appearance"), _
    Description("Get or Set the ShadowColor for selected tabs")> _
    Public Property ShadowColor_Selected() As Color
        Get
            Return _ShadowColor_Selected
        End Get
        Set(ByVal Value As Color)
            If Value <> _ShadowColor_Selected Then
                _ShadowColor_Selected = Value
                _Paint(Me.CreateGraphics, False)
            End If
        End Set
    End Property

    Private _ShadowColor_Unselected As Color = Color.Transparent
    <Category("Appearance"), _
    Description("Get or Set the ShadowColor for unselected tabs")> _
    Public Property ShadowColor_Unselected() As Color
        Get
            Return _ShadowColor_Unselected
        End Get
        Set(ByVal Value As Color)
            If Value <> _ShadowColor_Unselected Then
                _ShadowColor_Unselected = Value
                _Paint(Me.CreateGraphics, False)
            End If
        End Set
    End Property

    Private _BorderColor As Color = Color.WhiteSmoke
    <Category("Appearance"), _
    Description("Get or Set the BorderColor")> _
    Public Property BorderColor() As Color
        Get
            Return _BorderColor
        End Get
        Set(ByVal value As Color)
            If value <> _BorderColor Then
                _BorderColor = value
                _Paint(Me.CreateGraphics, False)
            End If
        End Set
    End Property



    Protected Overrides Sub OnPaint(ByVal e As System.Windows.Forms.PaintEventArgs)
        _Paint(e.Graphics, True)
    End Sub

    Private Sub _Paint(ByVal g As Graphics, ByVal clear As Boolean)
        '
        If Not Me.Visible Then Exit Sub
        '
        If clear Then g.Clear(_BackColor_Unselected)
        '
        If TabCount <= 0 Then Return
        '
        Dim rec, textrec As Rectangle
        Dim tp As TabPage
        Dim sf As StringFormat
        sf = New StringFormat
        sf.Alignment = StringAlignment.Center

        Dim LinePen As Pen
        LinePen = New Pen(_BorderColor)

        ' --------------------------------------------------------- Draw the Tabs
        For index As Integer = 0 To TabCount - 1
            tp = TabPages(index)
            rec = GetTabRect(index)
            textrec = rec

            '
            ' ----------------------------------------------------------------------- tab color and text
            If index = SelectedIndex Then
                ' ---------------------------------------------------------------------------------- selected tab
                g.FillRectangle(New System.Drawing.Drawing2D.LinearGradientBrush(rec, _
                                                                                 _BackColor_Selected1, _
                                                                                 _BackColor_Selected2, _
                                                                                 Drawing2D.LinearGradientMode.Vertical), _
                                                                                 rec)

                textrec.Offset(-2, 4)
                ' ----------------------------------------------------------------- draw the Shadow
                If Me._ShadowColor_Selected.A <> 0 Then
                    textrec.Offset(1, 1)
                    g.DrawString(tp.Text, tp.Font, _
                                          New SolidBrush(Me._ShadowColor_Selected), textrec, sf)
                    textrec.Offset(-1, -1)
                End If
                ' ------------------------------------------------------------------ draw the text
                g.DrawString(tp.Text, tp.Font, _
                                      New SolidBrush(_ForeColor_Selected), textrec, sf)
                '
                ' ----------------------------------------------------------------------------------- line left/upper/right
                g.DrawLine(LinePen, rec.Left, rec.Bottom, rec.Left, rec.Top)
                g.DrawLine(LinePen, rec.Left, rec.Top, rec.Right - 1, rec.Top)
                g.DrawLine(LinePen, rec.Right - 1, rec.Top, rec.Right - 1, rec.Bottom)

            Else
                ' ----------------------------------------------------------------------------------- unselected tab

                textrec.Offset(-2, 2)
                ' ----------------------------------------------------------------- draw the Shadow
                If Me._ShadowColor_Unselected.A <> 0 Then
                    textrec.Offset(1, 1)
                    g.DrawString(tp.Text, tp.Font, _
                                          New SolidBrush(Me._ShadowColor_Unselected), textrec, sf)
                    textrec.Offset(-1, -1)
                End If
                ' ------------------------------------------------------------------ draw the text
                g.DrawString(tp.Text, tp.Font, _
                                      New SolidBrush(_ForeColor_Unselected), textrec, sf)

                ' ----------------------------------------------------------------------------------- line bottom
                g.DrawLine(LinePen, rec.Left, rec.Bottom, rec.Right, rec.Bottom)
            End If
            '
            ' --------------------------------------------------------------------------------------- line bottom final
            rec = GetTabRect(TabCount - 1)
            g.DrawLine(LinePen, rec.Right, rec.Bottom, Right, rec.Bottom)
            '
        Next
    End Sub


End Class



