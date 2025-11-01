
Imports System.ComponentModel
Imports System.Windows.Forms
Imports System.Drawing

Public Class MyComboBox
    Inherits ComboBox


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
        'SetStyle(ControlStyles.AllPaintingInWmPaint Or _
        '   ControlStyles.DoubleBuffer Or _
        '   ControlStyles.ResizeRedraw Or _
        '   ControlStyles.SupportsTransparentBackColor Or _
        '   ControlStyles.UserPaint, True)

        'SetStyle(ControlStyles.AllPaintingInWmPaint, False)

        'SetStyle(ControlStyles.ResizeRedraw Or _
        '   ControlStyles.SupportsTransparentBackColor Or _
        '   ControlStyles.UserPaint, True)

        SetStyle(ControlStyles.AllPaintingInWmPaint Or _
           ControlStyles.DoubleBuffer Or _
           ControlStyles.ResizeRedraw Or _
           ControlStyles.SupportsTransparentBackColor, True)

        Me.FlatStyle = Windows.Forms.FlatStyle.Standard
        Me.DrawMode = Windows.Forms.DrawMode.OwnerDrawFixed
        Me.DropDownStyle = ComboBoxStyle.DropDownList
    End Sub




#Region "Properties"

    Private _BackColor_Focused As Color = Me.BackColor
    <Category("Appearance"), _
    Description("Get or Set the BackColor when the ComboBox has the focus")> _
    Public Property BackColor_Focused() As Color
        Get
            Return _BackColor_Focused
        End Get
        Set(ByVal Value As Color)
            _BackColor_Focused = Value
            Me.Invalidate()
        End Set
    End Property

    Private _BackColor_Over As Color = Me.BackColor
    <Category("Appearance"), _
    Description("Get or Set the BackColor when the mouse is Over the ComboBox")> _
    Public Property BackColor_Over() As Color
        Get
            Return _BackColor_Over
        End Get
        Set(ByVal Value As Color)
            _BackColor_Over = Value
            Me.Invalidate()
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
            _ShadowColor = Value
            Me.Invalidate()
        End Set
    End Property


    Private _ArrowColor As Color = Me.ForeColor
    <Category("Appearance"), _
    Description("Get or Set the ArrowColor")> _
    Public Property ArrowColor() As Color
        Get
            Return _ArrowColor
        End Get
        Set(ByVal Value As Color)
            _ArrowColor = Value
            Me.Invalidate()
        End Set
    End Property

    Private _ArrowButtonColor As Color = Me.BackColor
    <Category("Appearance"), _
    Description("Get or Set the ArrowButtonColor")> _
    Public Property ArrowButtonColor() As Color
        Get
            Return _ArrowButtonColor
        End Get
        Set(ByVal Value As Color)
            _ArrowButtonColor = Value
            Me.Invalidate()
        End Set
    End Property

    Private _BorderColor As Color = Me.BackColor
    <Category("Appearance"), _
    Description("Get or Set the BorderColor")> _
    Public Property BorderColor() As Color
        Get
            Return _BorderColor
        End Get
        Set(ByVal Value As Color)
            _BorderColor = Value
            Me.Invalidate()
        End Set
    End Property

    Private _BorderSize As Int32 = 1
    <Category("Appearance"), _
    Description("Get or Set the BorderSize")> _
    Public Property BorderSize() As Int32
        Get
            Return _BorderSize
        End Get
        Set(ByVal Value As Int32)
            If Value > 4 Then Value = 4
            If Value < 0 Then Value = 0
            _BorderSize = Value
            Me.Invalidate()
        End Set
    End Property

    Private _TextPosition As Int32 = 4
    <Category("Appearance"), _
    Description("Get or Set the Text vertical position")> _
    Public Property TextPosition() As Int32
        Get
            Return _TextPosition
        End Get
        Set(ByVal Value As Int32)
            If Value > 12 Then Value = 12
            If Value < 0 Then Value = 0
            _TextPosition = Value
            Me.Invalidate()
        End Set
    End Property


    Private _DropDown_BackColor As Color = Me.BackColor
    <Category("Appearance"), _
    Description("Get or Set the DropDown_BackColor")> _
    Public Property DropDown_BackColor() As Color
        Get
            Return _DropDown_BackColor
        End Get
        Set(ByVal Value As Color)
            _DropDown_BackColor = Value
            Me.Invalidate()
        End Set
    End Property

    Private _DropDown_BackSelected As Color = Me.BackColor
    <Category("Appearance"), _
    Description("Get or Set the DropDown_BackColor for selected items")> _
    Public Property DropDown_BackSelected() As Color
        Get
            Return _DropDown_BackSelected
        End Get
        Set(ByVal Value As Color)
            _DropDown_BackSelected = Value
            Me.Invalidate()
        End Set
    End Property

    Private _DropDown_ForeColor As Color = Me.ForeColor
    <Category("Appearance"), _
    Description("Get or Set the DropDown_ForeColor")> _
    Public Property DropDown_ForeColor() As Color
        Get
            Return _DropDown_ForeColor
        End Get
        Set(ByVal Value As Color)
            _DropDown_ForeColor = Value
            Me.Invalidate()
        End Set
    End Property

    Private _DropDown_ForeSelected As Color = Me.ForeColor
    <Category("Appearance"), _
    Description("Get or Set the DropDown_ForeColor for selected items")> _
    Public Property DropDown_ForeSelected() As Color
        Get
            Return _DropDown_ForeSelected
        End Get
        Set(ByVal Value As Color)
            _DropDown_ForeSelected = Value
            Me.Invalidate()
        End Set
    End Property

    Private _DropDown_BorderColor As Color = Me.BackColor
    <Category("Appearance"), _
    Description("Get or Set the DropDown_BorderColor")> _
    Public Property DropDown_BorderColor() As Color
        Get
            Return _DropDown_BorderColor
        End Get
        Set(ByVal Value As Color)
            _DropDown_BorderColor = Value
            Me.Invalidate()
        End Set
    End Property

#End Region 'Properties


#Region "Drawing"


    ' --------------------------------------------------------------------------------------------------------------
    '  LIST ITEMS COLORS
    ' --------------------------------------------------------------------------------------------------------------
    Private Sub MyComboBox_DrawItem(ByVal sender As Object, ByVal e As System.Windows.Forms.DrawItemEventArgs) Handles Me.DrawItem
        Dim g As Graphics = e.Graphics
        Dim r As Rectangle = e.Bounds

        If e.Index >= 0 Then


            If (e.State And DrawItemState.ComboBoxEdit) <> 0 Then

                ' =================
                '  EDIT AREA
                ' =================

                DrawEditArea(g)

            Else

                ' =================
                '  DROPDOWN AREA
                ' =================
                '

                ' --------------------------------------------------------------------- get the text that we wish to display
                Dim s As String = CType(Me.Items(e.Index), String)
                ' --------------------------------------------------------------------- Set the string format options
                Dim sf As StringFormat = New StringFormat
                sf.Alignment = StringAlignment.Near


                ' ----------------------------------------------------------------------- border
                Dim p As Pen = New Pen(DropDown_BorderColor, BorderSize * 2)
                e.Graphics.DrawLine(p, r.Left, r.Top - BorderSize, r.Left, r.Bottom + BorderSize)
                e.Graphics.DrawLine(p, r.Right, r.Top - BorderSize, r.Right, r.Bottom + BorderSize)
                '
                If e.Index = Me.Items.Count - 1 Then
                    e.Graphics.DrawLine(p, r.Left - BorderSize, r.Bottom, r.Right + BorderSize, r.Bottom)
                    r.Inflate(-BorderSize, -BorderSize \ 2)
                    r.Offset(0, -BorderSize)
                ElseIf e.Index = 0 Then
                    e.Graphics.DrawLine(p, r.Left - BorderSize, r.Top, r.Right + BorderSize, r.Top)
                    r.Inflate(-BorderSize, -BorderSize \ 2)
                    r.Offset(0, +BorderSize)
                Else
                    r.Inflate(-BorderSize, 0)
                End If


                'Debug.Print(e.State.ToString)

                If (e.State And DrawItemState.Focus) <> 0 Then
                    ' -------------------------------------------------------------------- background (selected)
                    e.Graphics.FillRectangle(New SolidBrush(_DropDown_BackSelected), r)
                    ' -------------------------------------------------------------------- text       (selected)
                    e.Graphics.DrawString(s, Me.Font, New SolidBrush(_DropDown_ForeSelected), r, sf)
                Else
                    ' -------------------------------------------------------------------- background (unselected)
                    e.Graphics.FillRectangle(New SolidBrush(_DropDown_BackColor), r)
                    ' -------------------------------------------------------------------- text       (unselected)
                    e.Graphics.DrawString(s, Me.Font, New SolidBrush(_DropDown_ForeColor), r, sf)
                End If

                e.DrawFocusRectangle()

            End If

        End If


    End Sub


    Protected Overrides Sub WndProc(ByRef m As Message)
        '
        MyBase.WndProc(m)
        '
        Const WM_PAINT As Int32 = 15
        Select Case m.Msg

            Case WM_PAINT
                DrawEditArea(Me.CreateGraphics())

        End Select
    End Sub


    Private Sub DrawEditArea(ByVal g As Graphics)
        Dim r As Rectangle

        ' ------------------------------------------------------------------ background borders (filled rectangle)
        Dim p As Pen = New Pen(Color.White, 1)
        g.FillRectangle(New SolidBrush(BorderColor), Me.ClientRectangle)

        ' ------------------------------------------------------------------ background of the text area
        r = Me.ClientRectangle
        r.Inflate(-_BorderSize, -_BorderSize)

        If _DrawState = eDrawState.Over Then
            g.FillRectangle(New SolidBrush(BackColor_Over), r)
        ElseIf Me.Focused Then
            g.FillRectangle(New SolidBrush(BackColor_Focused), r)
        Else
            g.FillRectangle(New SolidBrush(BackColor), r)
        End If

        ' ------------------------------------------------------------------ background of the dropdown button
        Dim rect As Rectangle = New Rectangle(Me.Width - 18, 3, 16, Me.Height - 6)
        g.FillRectangle(New SolidBrush(ArrowButtonColor), rect)

        ' ------------------------------------------------------------------ create the path for the arrow
        Dim pth As Drawing2D.GraphicsPath = New Drawing2D.GraphicsPath()
        Dim TopLeft As PointF = New PointF(Me.Width - 16, CSng(Me.Height - 5) / 2)
        Dim TopRight As PointF = New PointF(Me.Width - 4, CSng(Me.Height - 5) / 2)
        Dim Bottom As PointF = New PointF(Me.Width - 10, CSng(Me.Height + 6) / 2)
        pth.AddLine(TopLeft, TopRight)
        pth.AddLine(TopRight, Bottom)

        ' ------------------------------------------------------------------ draw the arrow
        g.SmoothingMode = Drawing2D.SmoothingMode.HighQuality
        g.FillPath(New SolidBrush(ArrowColor), pth)

        ' ------------------------------------------------------------------ draw the Shadow
        g.DrawString(Me.Text, Me.Font, New SolidBrush(Me._ShadowColor), 2, TextPosition + 1)

        ' ------------------------------------------------------------------ draw the text
        g.DrawString(Me.Text, Me.Font, New SolidBrush(Me.ForeColor), 1, TextPosition)
    End Sub

#End Region 'Drawing


#Region "Mouse"

    Private Enum eDrawState
        Over
        Normal
    End Enum
    Private _DrawState As eDrawState = eDrawState.Normal

    Protected Overrides Sub OnMouseEnter(ByVal e As System.EventArgs)
        MyBase.OnMouseEnter(e)
        _DrawState = eDrawState.Over
        Me.Invalidate()
    End Sub

    Protected Overrides Sub OnMouseLeave(ByVal e As System.EventArgs)
        MyBase.OnMouseLeave(e)
        _DrawState = eDrawState.Normal
        Me.Invalidate()
    End Sub

#End Region 'Mouse






End Class
