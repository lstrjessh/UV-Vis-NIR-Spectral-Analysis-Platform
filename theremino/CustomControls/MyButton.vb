Imports System.ComponentModel
Imports System.ComponentModel.Design
Imports System.Drawing.Design
Imports System.Drawing.Drawing2D
Imports System.Windows.Forms.Design
Imports System.Drawing.Imaging
Imports System.Drawing
Imports System.Windows.Forms


#Region "MyButton Class"

<ToolboxItem(True), ToolboxBitmap(GetType(MyButton), "MyButton.MyButton.bmp")> _
<DefaultEvent("ClickButtonArea")> _
<Designer(GetType(MyButtonDesigner))> _
Public Class MyButton
    Inherits Control

    Private DrawState As eDrawState = eDrawState.Up
    Private PressedOffset As Integer = 0
    Private Imagept As PointF
    Private ButtonArea As RectangleF
    Private TextArea As RectangleF
    Private ImageArea As RectangleF
    Private ImageSizeUse As Size

    Private _OverColorBlend As Color()
    Private _ClickColorBlend As Color()
    Private _OverColorBlendChecked As Color()
    Private _ClickColorBlendChecked As Color()
    Private _DisabledBlend As Color()

    Private _OverColorSolid As Color
    Private _OverColorSolidChecked As Color
    Private _ClickColorSolid As Color
    Private _ClickColorSolidChecked As Color
    Private _DisabledSolid As Color

    Public Sub New()

        ' This call is required by the Windows Form Designer.
        InitializeComponent()

        ' initializations
        SetStyle(ControlStyles.AllPaintingInWmPaint Or _
                ControlStyles.DoubleBuffer Or _
                ControlStyles.ResizeRedraw Or _
                ControlStyles.SupportsTransparentBackColor Or _
                ControlStyles.UserPaint, True)
        Me.Size = New Size(75, 25)
        UpdateDimColors()
        UpdateDimBlends()
    End Sub

#Region "Properties"

    Enum eDrawState
        Over
        Down
        Up
    End Enum

#Region "Shape"

#Region "Corners Expandable Property"

    'Corners Property is defined in the Corners Converter Class
    'to use the ExpandableObjectConverter to simulate the Padding Property

    Private _Corners As CornersProperty = New CornersProperty(6)

    <DesignerSerializationVisibility(DesignerSerializationVisibility.Content)> _
    <Category("Appearance MyButton"), _
      Description("Get or Set the Corner Radii")> _
    Public Property Corners() As CornersProperty
        Get
            Return _Corners
        End Get
        Set(ByVal Value As CornersProperty)
            _Corners = Value
            Me.Invalidate()
        End Set
    End Property

#End Region 'Corners Expandable Property

    Enum eShape
        Ellipse
        Rectangle
        TriangleUp
        TriangleDown
        TriangleLeft
        TriangleRight
    End Enum

    Private _Shape As eShape = eShape.Rectangle

    <Description("The Shape of the Button")> _
    <Category("Appearance MyButton")> _
    Public Property Shape() As eShape
        Get
            Return _Shape
        End Get
        Set(ByVal value As eShape)
            _Shape = value
            Me.Invalidate()
        End Set
    End Property

#End Region 'Shape

#Region "DimFactor"

    Private _DimFactorOver As Integer = 50
    <Category("Appearance MyButton"), _
   Description("Get or Set how much to dim the color on mouse rollover. Positive to Lighten and negative to Darken"), _
   DefaultValue(50)> _
    Public Property DimFactorOver() As Integer
        Get
            Return _DimFactorOver
        End Get
        Set(ByVal Value As Integer)
            _DimFactorOver = Value
            UpdateDimBlends()
            UpdateDimColors()
            Me.Invalidate()
        End Set
    End Property

    Private _DimFactorClick As Integer = -25
    <Category("Appearance MyButton"), _
   Description("Get or Set how much to dim the color on mouse down. Positive to Lighten and negative to Darken"), _
   DefaultValue(-25)> _
    Public Property DimFactorClick() As Integer
        Get
            Return _DimFactorClick
        End Get
        Set(ByVal Value As Integer)
            _DimFactorClick = Value
            UpdateDimBlends()
            UpdateDimColors()
            Me.Invalidate()
        End Set
    End Property

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
            UpdateDimBlends()
            UpdateDimColors()
            Me.Invalidate()
        End Set
    End Property
    Dim _DimGrayColors As Int32 = +40

#End Region 'DimFactor

#Region "Border"

    Private _BorderColor As Color = Color.Black
    <Category("Appearance MyButton"), _
   Description("Get or Set the Border color"), _
   DefaultValue(GetType(Color), "Black")> _
    Public Property BorderColor() As Color
        Get
            Return _BorderColor
        End Get
        Set(ByVal Value As Color)
            _BorderColor = Value
            Me.Invalidate()
        End Set
    End Property

    Private _BorderShow As Boolean = True
    <Category("Appearance MyButton"), _
   Description("Get or Set to show the Border"), _
   DefaultValue(True)> _
    Public Property BorderShow() As Boolean
        Get
            Return _BorderShow
        End Get
        Set(ByVal Value As Boolean)
            _BorderShow = Value
            Me.Invalidate()
        End Set
    End Property

#End Region 'Border

#Region "Fill"

    Enum eFillType
        Solid
        GradientPath
        LinearVertical
        LinearHorizontal
        LinearBackDiagonal
        LinearForwDiagonal
    End Enum

    Private _FillType As eFillType = eFillType.LinearVertical
    <Description("The Fill Type to apply to the MyButton")> _
    <Category("Appearance MyButton")> _
    Public Property FillType() As eFillType
        Get
            Return _FillType
        End Get
        Set(ByVal value As eFillType)
            _FillType = value
            Me.Invalidate()
        End Set
    End Property

    Private _FillTypeChecked As eFillType = eFillType.LinearHorizontal
    <Description("The Fill Type to apply to the MyButton when ""Checked""")> _
    <Category("Appearance MyButton")> _
    Public Property FillTypeChecked() As eFillType
        Get
            Return _FillTypeChecked
        End Get
        Set(ByVal value As eFillType)
            _FillTypeChecked = value
            Me.Invalidate()
        End Set
    End Property

    Private _ColorFillSolid As Color = SystemColors.Control
    <Description("The Solid Color to fill the MyButton"), _
    Category("Appearance MyButton")> _
    Public Property ColorFillSolid() As Color
        Get
            Return _ColorFillSolid
        End Get
        Set(ByVal value As Color)
            _ColorFillSolid = value
            UpdateDimColors()
            Me.Invalidate()
        End Set
    End Property

    Private _ColorFillSolidChecked As Color = SystemColors.Control
    <Description("The Solid Color to fill the MyButton when ""Checked"""), _
    Category("Appearance MyButton")> _
    Public Property ColorFillSolidChecked() As Color
        Get
            Return _ColorFillSolidChecked
        End Get
        Set(ByVal value As Color)
            _ColorFillSolidChecked = value
            UpdateDimColors()
            Me.Invalidate()
        End Set
    End Property

    Private _ColorFillBlend As cBlendItems = New cBlendItems(New Color() {Color.AliceBlue, Color.RoyalBlue, Color.Navy}, New Single() {0, 0.5, 1})
    <Description("The ColorBlend to fill the MyButton"), _
    Category("Appearance MyButton"), _
    Editor(GetType(BlendTypeEditor), GetType(UITypeEditor))> _
    Public Property ColorFillBlend() As cBlendItems
        Get
            Return _ColorFillBlend
        End Get
        Set(ByVal value As cBlendItems)
            _ColorFillBlend = value
            UpdateDimBlends()
            Me.Invalidate()
        End Set
    End Property

    Private _ColorFillBlendChecked As cBlendItems = New cBlendItems(New Color() {Color.Red, Color.Orange, Color.Orange}, New Single() {0, 0.5, 1})
    <Description("The ColorBlend to fill the MyButton when ""Checked"""), _
    Category("Appearance MyButton"), _
    Editor(GetType(BlendTypeEditor), GetType(UITypeEditor))> _
    Public Property ColorFillBlendChecked() As cBlendItems
        Get
            Return _ColorFillBlendChecked
        End Get
        Set(ByVal value As cBlendItems)
            _ColorFillBlendChecked = value
            UpdateDimBlends()
            Me.Invalidate()
        End Set
    End Property


#End Region 'Fill

#Region "Text"

    Private _Text As String = "MyButton"
    <Category("Appearance MyButton"), _
   Description("Get or Set the Button Text"), _
   DefaultValue("MyButton")> _
   Overrides Property Text() As String
        Get
            Return _Text
        End Get
        Set(ByVal value As String)
            If value <> _Text Then
                _Text = value
                Me.Invalidate()
            End If
        End Set
    End Property

    Private _TextAlign As ContentAlignment = ContentAlignment.MiddleCenter
    <Category("Appearance MyButton"), _
   Description("Get or Set the alignment for the text"), _
   DefaultValue(ContentAlignment.MiddleCenter)> _
    Public Property TextAlign() As ContentAlignment
        Get
            Return _TextAlign
        End Get
        Set(ByVal Value As ContentAlignment)
            _TextAlign = Value
            Me.Invalidate()
        End Set
    End Property

    Private _TextMargin As Padding
    <Category("Appearance MyButton"), _
   Description("Get or Set the margion between the text and the button edge"), _
   DefaultValue("2,2,2,2")> _
    Public Property TextMargin() As Padding
        Get
            Return _TextMargin
        End Get
        Set(ByVal Value As Padding)
            _TextMargin = Value
            Me.Invalidate()
        End Set
    End Property

    Private _TextImageRelation As TextImageRelation = TextImageRelation.Overlay
    <Category("Appearance MyButton"), _
   Description("Get or Set the Relationship of the Text to the Image"), _
   DefaultValue("(None)")> _
    Public Property TextImageRelation() As TextImageRelation
        Get
            Return _TextImageRelation
        End Get
        Set(ByVal Value As TextImageRelation)
            _TextImageRelation = Value
            Me.Invalidate()
        End Set
    End Property


    Private _ForeColorChecked As Color = Color.DimGray
    <Category("Appearance"), _
   Description("Get or Set the color of the Text (Checked button)"), _
   DefaultValue(GetType(Color), "DimGray")> _
    Public Property ForeColorChecked() As Color
        Get
            Return _ForeColorChecked
        End Get
        Set(ByVal Value As Color)
            _ForeColorChecked = Value
            Me.Invalidate()
        End Set
    End Property

    Private _TextShadow As Color = Color.Transparent
    <Category("Appearance MyButton"), _
   Description("Get or Set the color of the Shadow ( transparent = no shadow )"), _
   DefaultValue(GetType(Color), "DimGray")> _
    Public Property TextShadow() As Color
        Get
            Return _TextShadow
        End Get
        Set(ByVal Value As Color)
            _TextShadow = Value
            Me.Invalidate()
        End Set
    End Property

    Private _TextShadowChecked As Color = Color.DimGray
    <Category("Appearance MyButton"), _
   Description("Get or Set the color of the Shadow Text (Checked Button)"), _
   DefaultValue(GetType(Color), "DimGray")> _
    Public Property TextShadowChecked() As Color
        Get
            Return _TextShadowChecked
        End Get
        Set(ByVal Value As Color)
            _TextShadowChecked = Value
            Me.Invalidate()
        End Set
    End Property

#End Region 'Text

#Region "Image"

    Private _Image As Image = Nothing
    <Category("Appearance MyButton"), _
   Description("Get or Set the small Image next to text"), _
   DefaultValue("(None)")> _
    Public Property Image() As Image
        Get
            Return _Image
        End Get
        Set(ByVal Value As Image)
            _Image = Value
            Me.Invalidate()
        End Set
    End Property

    Private _ImageAlign As ContentAlignment = ContentAlignment.MiddleCenter
    <Category("Appearance MyButton"), _
   Description("Get or Set the placement of the Image"), _
   DefaultValue("(None)")> _
    Public Property ImageAlign() As ContentAlignment
        Get
            Return _ImageAlign
        End Get
        Set(ByVal Value As ContentAlignment)
            _ImageAlign = Value
            Me.Invalidate()
        End Set
    End Property

    Private _ImageSize As Size = New Size(16, 16)
    <Category("Appearance MyButton"), _
   Description("Get or Set the Size of the Image"), _
   DefaultValue("16, 16")> _
    Public Property ImageSize() As Size
        Get
            Return _ImageSize
        End Get
        Set(ByVal Value As Size)
            _ImageSize = Value
            Me.Invalidate()
        End Set
    End Property

    Private _Imagelist As New ImageList
    <Category("Appearance MyButton"), _
   Description("Get or Set the ImageList control"), _
   DefaultValue("(None)")> _
    Public Property Imagelist() As ImageList
        Get
            Return _Imagelist
        End Get
        Set(ByVal Value As ImageList)
            _Imagelist = Value
            Me.Invalidate()
        End Set
    End Property

    Private _ImageIndex As Integer
    <Category("Appearance MyButton"), _
   Description("Get or Set the ImageList control")> _
    Public Property ImageIndex() As Integer
        Get
            Return _ImageIndex
        End Get
        Set(ByVal Value As Integer)
            If Imagelist.Images.Count > 0 Then
                If Value >= 0 And Value < Imagelist.Images.Count Then
                    _ImageIndex = Value
                    Me.Image = Imagelist.Images.Item(Value)
                    Me.Invalidate()
                End If
            End If
        End Set
    End Property

#End Region 'Image

#Region "SideImage"

    Private _SideImage As Image = Nothing
    <Category("Appearance MyButton"), _
   Description("Get or Set the Side Image"), _
   DefaultValue("(None)")> _
    Public Property SideImage() As Image
        Get
            Return _SideImage
        End Get
        Set(ByVal Value As Image)
            _SideImage = Value
            If Not Value Is Nothing Then
                _SideImageSize = Value.Size
            End If
            Me.Invalidate()
        End Set
    End Property

    Private _SideImageSize As Size = New Size(48, 48)
    <Category("Appearance MyButton"), _
   Description("Get or Set the Size of the Side Image"), _
   DefaultValue("48, 48")> _
    Public Property SideImageSize() As Size
        Get
            Return _SideImageSize
        End Get
        Set(ByVal Value As Size)
            _SideImageSize = Value
            Me.Invalidate()
        End Set
    End Property

    Private _SideImageBehindText As Boolean = True
    <Category("Appearance MyButton"), _
   Description("Get or Set if the Side Image is in front or behind the Text"), _
   DefaultValue(True)> _
    Public Property SideImageBehindText() As Boolean
        Get
            Return _SideImageBehindText
        End Get
        Set(ByVal Value As Boolean)
            _SideImageBehindText = Value
            Me.Invalidate()
        End Set
    End Property

    Private _SideImageAlign As ContentAlignment = ContentAlignment.MiddleLeft
    <Category("Appearance MyButton"), _
   Description("Get or Set the Side Image Alignment"), _
   DefaultValue(ContentAlignment.MiddleLeft)> _
    Public Property SideImageAlign() As ContentAlignment
        Get
            Return _SideImageAlign
        End Get
        Set(ByVal Value As ContentAlignment)
            _SideImageAlign = Value
            Me.Invalidate()
        End Set
    End Property

#End Region 'SideImage

#Region "FocalPoints"

    Private _FocalPoints As cFocalPoints = New cFocalPoints(0.5, 0.5, 0, 0)

    <DesignerSerializationVisibility(DesignerSerializationVisibility.Content)> _
    <Description("The CenterPoint and FocusScales for the ColorBlend"), _
    Category("Appearance MyButton")> _
    Public Property FocalPoints() As cFocalPoints
        Get
            Return _FocalPoints
        End Get
        Set(ByVal value As cFocalPoints)
            _FocalPoints = value
            CenterPtTracker.TrackerRectangle = CenterPtTrackerRectangle()
            FocusPtTracker.TrackerRectangle = FocusPtTrackerRectangle()
            Me.Invalidate()
        End Set
    End Property


    Private _FocalPoints_Checked As cFocalPoints = New cFocalPoints(0.5, 0.5, 0, 0)

    <DesignerSerializationVisibility(DesignerSerializationVisibility.Content)> _
    <Description("The CenterPoint and FocusScales for the ColorBlend when ""Checked"""), _
    Category("Appearance MyButton")> _
    Public Property FocalPointsChecked() As cFocalPoints
        Get
            Return _FocalPoints_Checked
        End Get
        Set(ByVal value As cFocalPoints)
            _FocalPoints_Checked = value
            CenterPtTracker.TrackerRectangle = CenterPtTrackerRectangle()
            FocusPtTracker.TrackerRectangle = FocusPtTrackerRectangle()
            Me.Invalidate()
        End Set
    End Property


    Private Function CenterPtTrackerRectangle() As RectangleF
        If Checked Then
            Return New RectangleF((Me.FocalPointsChecked.CenterPoint.X * Me.Width) - 5, _
                                  (Me.FocalPointsChecked.CenterPoint.Y * Me.Height) - 5, 10, 10)
        Else
            Return New RectangleF((Me.FocalPoints.CenterPoint.X * Me.Width) - 5, _
                                  (Me.FocalPoints.CenterPoint.Y * Me.Height) - 5, 10, 10)
        End If
    End Function

    Private Function FocusPtTrackerRectangle() As RectangleF
        If Checked Then
            Return New RectangleF((Me.FocalPointsChecked.FocusScales.X * Me.Width) - 5, _
                                  (Me.FocalPointsChecked.FocusScales.Y * Me.Height) - 5, 10, 10)
        Else
            Return New RectangleF((Me.FocalPoints.FocusScales.X * Me.Width) - 5, _
                                  (Me.FocalPoints.FocusScales.Y * Me.Height) - 5, 10, 10)
        End If
    End Function

    Private _CenterPtTracker As New DesignerRectTracker

    <Browsable(False)> _
    <Category("Shape")> _
    Public Property CenterPtTracker() As DesignerRectTracker
        Get
            Return _CenterPtTracker
        End Get
        Set(ByVal value As DesignerRectTracker)
            _CenterPtTracker = value
        End Set
    End Property

    Private _FocusPtTracker As New DesignerRectTracker

    <Browsable(False)> _
    <Category("Shape")> _
    Public Property FocusPtTracker() As DesignerRectTracker
        Get
            Return _FocusPtTracker
        End Get
        Set(ByVal value As DesignerRectTracker)
            _FocusPtTracker = value
        End Set
    End Property

#End Region 'FocalPoints

#Region "CheckButton"

    Private _CheckButton As Boolean = False
    '<DesignerSerializationVisibility(DesignerSerializationVisibility.Content)> _
    <Description("Get or Set to enable ""Check button"" behavior"), _
    Category("Appearance MyButton"), _
    DefaultValue(False)> _
    Public Property CheckButton() As Boolean
        Get
            Return _CheckButton
        End Get
        Set(ByVal value As Boolean)
            If value <> _CheckButton Then
                _CheckButton = value
                Me.Invalidate()
            End If
        End Set
    End Property

    Private _Checked As Boolean = False
    '<DesignerSerializationVisibility(DesignerSerializationVisibility.Content)> _
    <Description("Get or Set the ""Checked"" state"), _
    Category("Appearance MyButton"), _
    DefaultValue(False)> _
    Public Property Checked() As Boolean
        Get
            Return _Checked
        End Get
        Set(ByVal value As Boolean)
            If value <> _Checked Then
                _Checked = value
                Me.Invalidate()
            End If
        End Set
    End Property

#End Region 'CheckButton


#Region "AutoRepeat"

    Private WithEvents _timer As System.Windows.Forms.Timer = Nothing

    Private _AutoRepeat As Boolean = False
    <Description("Get or Set the ""AutoRepeat"" state"), _
    Category("AutoRepeat"), _
    DefaultValue(False)> _
    Public Property AutoRepeat() As Boolean
        Get
            Return _AutoRepeat
        End Get
        Set(ByVal value As Boolean)
            _AutoRepeat = value
            If _AutoRepeat Then
                _timer = New System.Windows.Forms.Timer
            Else
                _timer = Nothing
            End If
        End Set
    End Property

    Private _Interval As Int32 = 400
    <Description("Get or Set the Repeat Interval in milliseconds"), _
    Category("AutoRepeat"), _
    DefaultValue(400)> _
    Public Property Interval() As Int32
        Get
            Return _Interval
        End Get
        Set(ByVal value As Int32)
            If value < 1 Then value = 1
            _Interval = value
        End Set
    End Property

    Private _Interval_Min As Int32 = 100
    <Description("Get or Set the Minimum Repeat Interval in milliseconds"), _
    Category("AutoRepeat"), _
    DefaultValue(100)> _
    Public Property Interval_Min() As Int32
        Get
            Return _Interval_Min
        End Get
        Set(ByVal value As Int32)
            If value < 1 Then value = 1
            _Interval_Min = value
        End Set
    End Property

    Private _Interval_Decrement As Int32 = 50
    <Description("Get or Set the Repeat Interval decrement in milliseconds"), _
    Category("AutoRepeat"), _
    DefaultValue(50)> _
    Public Property Interval_Decrement() As Int32
        Get
            Return _Interval_Decrement
        End Get
        Set(ByVal value As Int32)
            _Interval_Decrement = value
        End Set
    End Property

    Public Event AutoRepeatClick(ByVal Sender As Object, ByVal e As EventArgs)

    Private Sub OnTimer(ByVal sender As Object, ByVal e As System.EventArgs) Handles _timer.Tick
        RaiseEvent AutoRepeatClick(Me, New EventArgs)
        If _timer.Interval - _Interval_Decrement >= _Interval_Min Then
            _timer.Interval -= _Interval_Decrement
        Else
            _timer.Interval = _Interval_Min
        End If
    End Sub
    Private Sub RepeatButton_MouseDown(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles MyBase.MouseDown
        If _timer Is Nothing Then Exit Sub
        RaiseEvent AutoRepeatClick(Me, New EventArgs)
        _timer.Interval = _Interval
        _timer.Enabled = True
    End Sub
    Private Sub RepeatButton_MouseUp(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles MyBase.MouseUp
        If _timer Is Nothing Then Exit Sub
        _timer.Enabled = False
    End Sub

#End Region 'AutoRepeat

#End Region 'Properties

#Region "Drawing"

    Protected Overrides Sub OnPaint(ByVal e As System.Windows.Forms.PaintEventArgs)

        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias

        'Gray the Text and Border Colors if Disabled
        Dim bColor, tColor, tsColor As Color
        If Me.Enabled Then
            bColor = _BorderColor
            If _Checked Then
                tColor = Me.ForeColorChecked
                tsColor = _TextShadowChecked
            Else
                tColor = Me.ForeColor
                tsColor = _TextShadow
            End If
        Else
            bColor = GrayTheColor(_BorderColor)
            tColor = GrayTheColor(Me.ForeColor)
            tsColor = GrayTheColor(_TextShadow)
        End If

        Dim MyPen As New Pen(bColor)

        If OperatingSystemIsWindows Then
            MyPen.Alignment = PenAlignment.Inset
        End If

        'Shrink the Area so the Border draws correctly, _
        'then trim off the Padding to get the button surface area
        ButtonArea = AdjustRect(New RectangleF(0, 0, Me.Size.Width - 1, Me.Size.Height - 1), Me.Padding)

        'Create the ButtonArea Path
        Dim gp As GraphicsPath = GetPath()

        If Me.BackgroundImage Is Nothing Then

            'Color the ButtonArea with the right Brush
            Dim ft As eFillType
            If _Checked And Enabled Then
                ft = Me.FillTypeChecked
            Else
                ft = Me.FillType
            End If
            Select Case ft
                Case eFillType.Solid
                    Using br As Brush = New SolidBrush(GetFill)
                        e.Graphics.FillPath(br, gp)
                    End Using

                Case eFillType.GradientPath
                    Using br As PathGradientBrush = New PathGradientBrush(gp)

                        Dim cb As New ColorBlend
                        cb.Colors = GetFillBlend()
                        cb.Positions = GetFillPositions()
                        If Checked Then
                            br.FocusScales = FocalPointsChecked.FocusScales
                            br.CenterPoint = New PointF(Me.Width * FocalPointsChecked.CenterPoint.X, _
                                                        Me.Height * FocalPointsChecked.CenterPoint.Y)
                        Else
                            br.FocusScales = FocalPoints.FocusScales
                            br.CenterPoint = New PointF(Me.Width * FocalPoints.CenterPoint.X, _
                                                        Me.Height * FocalPoints.CenterPoint.Y)
                        End If
                        br.InterpolationColors = cb
                        e.Graphics.FillPath(br, gp)
                    End Using

                Case eFillType.LinearHorizontal
                    Using br As LinearGradientBrush = New LinearGradientBrush(ButtonArea, _
                                                                              Color.White, _
                                                                              Color.White, _
                                                                              LinearGradientMode.Horizontal)
                        Dim cb As New ColorBlend
                        cb.Colors = GetFillBlend()
                        cb.Positions = GetFillPositions()
                        br.InterpolationColors = cb
                        e.Graphics.FillPath(br, gp)
                    End Using


                Case eFillType.LinearVertical
                    Using br As LinearGradientBrush = New LinearGradientBrush(ButtonArea, _
                                                                              Color.White, _
                                                                              Color.White, _
                                                                              LinearGradientMode.Vertical)
                        Dim cb As New ColorBlend
                        cb.Colors = GetFillBlend()
                        cb.Positions = GetFillPositions()
                        br.InterpolationColors = cb
                        e.Graphics.FillPath(br, gp)
                    End Using

                Case eFillType.LinearForwDiagonal
                    Using br As LinearGradientBrush = New LinearGradientBrush(ButtonArea, _
                                                                              Color.White, _
                                                                              Color.White, _
                                                                               LinearGradientMode.ForwardDiagonal)
                        Dim cb As New ColorBlend
                        cb.Colors = GetFillBlend()
                        cb.Positions = GetFillPositions()
                        br.InterpolationColors = cb
                        e.Graphics.FillPath(br, gp)
                    End Using

                Case eFillType.LinearBackDiagonal
                    Using br As LinearGradientBrush = New LinearGradientBrush(ButtonArea, _
                                                                              Color.White, _
                                                                              Color.White, _
                                                                               LinearGradientMode.BackwardDiagonal)
                        Dim cb As New ColorBlend
                        cb.Colors = GetFillBlend()
                        cb.Positions = GetFillPositions()
                        br.InterpolationColors = cb
                        e.Graphics.FillPath(br, gp)
                    End Using

            End Select
        End If

        If BorderShow Then
            e.Graphics.DrawPath(MyPen, gp)
        End If
        gp.Dispose()

        Dim ipt As PointF = ImageLocation(GetStringFormat(Me.SideImageAlign), _
                                            Me.Size, _
                                            Me.SideImageSize)

        'Put the SideImage behind the Text
        If SideImageBehindText AndAlso Me.SideImage IsNot Nothing Then
            e.Graphics.DrawImage(EnableDisableImage(Me.SideImage), _
                                 ipt.X, ipt.Y, _
                                 Me.SideImageSize.Width, _
                                 Me.SideImageSize.Height)
        End If

        'Layout the Text and Image on the button surface
        SetImageAndText(e.Graphics)

        If Me.Image IsNot Nothing Then
            e.Graphics.DrawImage(EnableDisableImage(Me.Image), _
                                 Imagept.X, Imagept.Y, _
                                 Me.ImageSize.Width, _
                                 Me.ImageSize.Height)
        End If

        ' ----------------------------------------------------------------- draw the Shadow
        If tsColor.A <> 0 Then
            TextArea.Offset(1, 1)
            e.Graphics.DrawString(Me.Text, Me.Font, _
                                  New SolidBrush(tsColor), TextArea, GetStringFormat(Me.TextAlign))
            TextArea.Offset(-1, -1)
        End If
        ' ----------------------------------------------------------------- draw the Text
        e.Graphics.DrawString(Me.Text, Me.Font, _
                              New SolidBrush(tColor), TextArea, GetStringFormat(Me.TextAlign))


        ' ----------------------------------------------------------------- put the SideImage in front of the Text
        If Not SideImageBehindText AndAlso Me.SideImage IsNot Nothing Then
            e.Graphics.DrawImage(EnableDisableImage(Me.SideImage), ipt.X, ipt.Y, _
                                 Me.SideImageSize.Width, Me.SideImageSize.Height)
        End If

        MyPen.Dispose()
    End Sub

    Private Function EnableDisableImage(ByVal img As Image) As Bitmap
        If Me.Enabled Then Return CType(img, Bitmap)
        Dim bm As Bitmap = New Bitmap(img.Width, img.Height)
        Dim g As Graphics = Graphics.FromImage(bm)

        Dim kr As Single = CSng(0.299 * (100 + _DimFactorGray) / 100)
        Dim kg As Single = CSng(0.587 * (100 + _DimFactorGray) / 100)
        Dim kb As Single = CSng(0.114 * (100 + _DimFactorGray) / 100)

        Dim cm As ColorMatrix = New ColorMatrix(New Single()() _
                                                 {New Single() {kr, kr, kr, 0, 0}, _
                                                  New Single() {kg, kg, kg, 0, 0}, _
                                                  New Single() {kb, kb, kb, 0, 0}, _
                                                  New Single() {0, 0, 0, 1, 0}, _
                                                  New Single() {0, 0, 0, 0, 1}})

        Dim ia As ImageAttributes = New ImageAttributes()
        ia.SetColorMatrix(cm)
        g.DrawImage(img, New Rectangle(0, 0, img.Width, img.Height), 0, 0, img.Width, img.Height, GraphicsUnit.Pixel, ia)
        g.Dispose()
        Return bm
    End Function

    Private Function GetPath() As GraphicsPath
        Dim gp As New GraphicsPath

        Select Case _Shape

            Case eShape.Ellipse
                gp.AddEllipse(ButtonArea)

            Case eShape.Rectangle
                gp = GetRoundedRectPath(ButtonArea)

            Case eShape.TriangleUp
                Dim pts() As PointF = New PointF() { _
                    New PointF(CSng(ButtonArea.Width / 2), ButtonArea.Y), _
                    New PointF(ButtonArea.Width, ButtonArea.Y + ButtonArea.Height), _
                    New PointF(ButtonArea.X, ButtonArea.Y + ButtonArea.Height)}
                gp.AddPolygon(pts)

            Case eShape.TriangleDown
                Dim pts() As PointF = New PointF() { _
                    New PointF(ButtonArea.X, ButtonArea.Y), _
                    New PointF(CSng(ButtonArea.Width / 2), ButtonArea.Y + ButtonArea.Height), _
                    New PointF(ButtonArea.X + ButtonArea.Width, ButtonArea.Y)}
                gp.AddPolygon(pts)

            Case eShape.TriangleLeft
                Dim pts() As PointF = New PointF() { _
                    New PointF(ButtonArea.X, CSng(ButtonArea.Y + (ButtonArea.Height / 2))), _
                    New PointF(ButtonArea.Width, ButtonArea.Y), _
                    New PointF(ButtonArea.Width, ButtonArea.Y + ButtonArea.Height)}
                gp.AddPolygon(pts)

            Case eShape.TriangleRight
                Dim pts() As PointF = New PointF() { _
                    New PointF(ButtonArea.X, ButtonArea.Y), _
                    New PointF(ButtonArea.Width, CSng(ButtonArea.Y + (ButtonArea.Height / 2))), _
                    New PointF(ButtonArea.X, ButtonArea.Y + ButtonArea.Height)}
                gp.AddPolygon(pts)

        End Select

        Return gp
    End Function

    Private Function GetStringFormat(ByVal ctrlalign As ContentAlignment) As StringFormat
        Dim strFormat As StringFormat = New StringFormat()
        Select Case ctrlalign
            Case ContentAlignment.MiddleCenter
                strFormat.LineAlignment = StringAlignment.Center
                strFormat.Alignment = StringAlignment.Center
            Case ContentAlignment.MiddleLeft
                strFormat.LineAlignment = StringAlignment.Center
                strFormat.Alignment = StringAlignment.Near
            Case ContentAlignment.MiddleRight
                strFormat.LineAlignment = StringAlignment.Center
                strFormat.Alignment = StringAlignment.Far
            Case ContentAlignment.TopCenter
                strFormat.LineAlignment = StringAlignment.Near
                strFormat.Alignment = StringAlignment.Center
            Case ContentAlignment.TopLeft
                strFormat.LineAlignment = StringAlignment.Near
                strFormat.Alignment = StringAlignment.Near
            Case ContentAlignment.TopRight
                strFormat.LineAlignment = StringAlignment.Near
                strFormat.Alignment = StringAlignment.Far
            Case ContentAlignment.BottomCenter
                strFormat.LineAlignment = StringAlignment.Far
                strFormat.Alignment = StringAlignment.Center
            Case ContentAlignment.BottomLeft
                strFormat.LineAlignment = StringAlignment.Far
                strFormat.Alignment = StringAlignment.Near
            Case ContentAlignment.BottomRight
                strFormat.LineAlignment = StringAlignment.Far
                strFormat.Alignment = StringAlignment.Far
        End Select
        Return strFormat
    End Function

    Sub SetImageAndText(ByVal g As Graphics)
        PressedOffset = 0
        If DrawState = eDrawState.Down Then PressedOffset = 1

        If Not Me.Image Is Nothing Then
            ImageSizeUse = Me.ImageSize
        Else
            ImageSizeUse = New Size(0, 0)
        End If

        Select Case Me.TextImageRelation
            Case Windows.Forms.TextImageRelation.Overlay, Windows.Forms.TextImageRelation.ImageAboveText, Windows.Forms.TextImageRelation.TextAboveImage
                TextArea = AdjustRect(ButtonArea, TextMargin)
                ImageArea = ButtonArea
                TextArea.Y += PressedOffset
                Imagept = ImageLocation(GetStringFormat(Me.ImageAlign), ButtonArea.Size, ImageSizeUse)
                Imagept.X += ButtonArea.X
            Case Windows.Forms.TextImageRelation.ImageBeforeText
                Dim TextSize As SizeF = g.MeasureString(Me.Text, Me.Font)
                TextArea = AdjustRect(ButtonArea, TextMargin)
                TextArea.Width -= ImageSizeUse.Width - 4
                TextArea.Y += PressedOffset
                ImageArea = New RectangleF(TextArea.X - ImageSizeUse.Width, ButtonArea.Y, ImageSizeUse.Width, ImageSizeUse.Height)
                Imagept = ImageLocation(GetStringFormat(Me.ImageAlign), ButtonArea.Size, ImageArea.Size)

                Select Case GetStringFormat(Me.TextAlign).Alignment
                    Case StringAlignment.Center
                        Imagept.X = ButtonArea.X + ((ButtonArea.Width - TextSize.Width - ImageSizeUse.Width) / 2)
                        TextArea.X = ButtonArea.X + ImageSizeUse.Width
                    Case StringAlignment.Near
                        Imagept.X = ButtonArea.X + 4
                        TextArea.X = ButtonArea.X + ImageSizeUse.Width + 4
                    Case StringAlignment.Far
                        Imagept.X = ButtonArea.X + TextArea.Width - TextSize.Width - 12
                        TextArea.X = ButtonArea.X + ImageSizeUse.Width - 8
                End Select

            Case Windows.Forms.TextImageRelation.TextBeforeImage
                Dim TextSize As SizeF = g.MeasureString(Me.Text, Me.Font)
                TextArea = AdjustRect(ButtonArea, TextMargin)
                TextArea.Width -= ImageSizeUse.Width - 8
                TextArea.Y += PressedOffset
                ImageArea = New RectangleF(TextArea.X, ButtonArea.Y, ImageSizeUse.Width, ImageSizeUse.Height)
                Imagept = ImageLocation(GetStringFormat(Me.ImageAlign), ButtonArea.Size, ImageArea.Size)

                Select Case GetStringFormat(Me.TextAlign).Alignment
                    Case StringAlignment.Center
                        Imagept.X = ((TextArea.Width - TextSize.Width) / 2) + TextSize.Width
                        TextArea.X = -4
                    Case StringAlignment.Near
                        Imagept.X = TextSize.Width + 8
                        TextArea.X = 4
                    Case StringAlignment.Far
                        Imagept.X = TextArea.Width - 12
                        TextArea.X = -16
                End Select

        End Select
        Imagept.Y += PressedOffset + ButtonArea.Y

    End Sub

    Function ImageLocation(ByVal sf As StringFormat, ByVal Area As SizeF, ByVal ImageArea As SizeF) As PointF
        Dim pt As PointF
        '
        Select Case sf.Alignment
            Case StringAlignment.Center
                pt.X = CSng((Area.Width - ImageArea.Width) / 2)
            Case StringAlignment.Near
                pt.X = 2
            Case StringAlignment.Far
                pt.X = Area.Width - ImageArea.Width - 2
        End Select
        '
        Select Case sf.LineAlignment
            Case StringAlignment.Center
                pt.Y = CSng((Area.Height - ImageArea.Height) / 2)
            Case StringAlignment.Near
                pt.Y = 2
            Case StringAlignment.Far
                pt.Y = Area.Height - ImageArea.Height - 2
        End Select
        '
        Return pt
    End Function

    Private Sub UpdateDimBlends()
        Dim ca As Color()
        Dim c As Color
        '
        ca = _ColorFillBlend.iColor
        _OverColorBlend = CType(ca.Clone, Color())
        _ClickColorBlend = CType(ca.Clone, Color())
        _DisabledBlend = CType(ca.Clone, Color())
        For i As Integer = 0 To _ColorFillBlend.iColor.Length - 1
            c = ca(i)
            _OverColorBlend(i) = DimTheColor(c, _DimFactorOver)
            _ClickColorBlend(i) = DimTheColor(c, _DimFactorClick)
            _DisabledBlend(i) = GrayTheColor(c)
        Next
        '
        ca = _ColorFillBlendChecked.iColor
        _OverColorBlendChecked = CType(ca.Clone, Color())
        _ClickColorBlendChecked = CType(ca.Clone, Color())
        For i As Integer = 0 To _ColorFillBlendChecked.iColor.Length - 1
            c = ca(i)
            _OverColorBlendChecked(i) = DimTheColor(c, _DimFactorOver)
            _ClickColorBlendChecked(i) = DimTheColor(c, _DimFactorClick)
        Next
    End Sub

    Private Sub UpdateDimColors()
        _OverColorSolid = DimTheColor(_ColorFillSolid, _DimFactorOver)
        _ClickColorSolid = DimTheColor(_ColorFillSolid, _DimFactorClick)
        _OverColorSolidChecked = DimTheColor(_ColorFillSolidChecked, _DimFactorOver)
        _ClickColorSolidChecked = DimTheColor(_ColorFillSolidChecked, _DimFactorClick)
        _DisabledSolid = GrayTheColor(_ColorFillSolid)
    End Sub

    Function DimTheColor(ByVal DimColor As Color, ByVal DimDegree As Integer) As Color
        '
        ' --------------------------------------------------- additive dimming
        Dim ColorR As Integer = DimColor.R + DimDegree
        Dim ColorG As Integer = DimColor.G + DimDegree
        Dim ColorB As Integer = DimColor.B + DimDegree
        '
        ' --------------------------------------------------- percentual dimming
        'Dim kdim As Double = (100 + DimDegree) / 100
        'Dim ColorR As Integer = CInt(DimColor.R * kdim)
        'Dim ColorG As Integer = CInt(DimColor.G * kdim)
        'Dim ColorB As Integer = CInt(DimColor.B * kdim)
        '
        If ColorR > 255 Then ColorR = 255
        If ColorG > 255 Then ColorG = 255
        If ColorB > 255 Then ColorB = 255
        If ColorR < 0 Then ColorR = 0
        If ColorG < 0 Then ColorG = 0
        If ColorB < 0 Then ColorB = 0
        '
        Return Color.FromArgb(DimColor.A, ColorR, ColorG, ColorB)
    End Function

    Private Function GrayTheColor(ByVal GrayColor As Color) As Color
        Dim gray As Integer = CInt(GrayColor.R * 0.3 + GrayColor.G * 0.59 + GrayColor.B * 0.11)
        gray = CInt(gray * (100 + _DimFactorGray) / 100)
        If gray < 0 Then gray = 0
        If gray > 255 Then gray = 255
        Return Color.FromArgb(GrayColor.A, gray, gray, gray)
    End Function

    Private Function GetFillPositions() As Single()
        If _Checked And Me.Enabled Then
            Return Me.ColorFillBlendChecked.iPoint
        Else
            Return Me.ColorFillBlend.iPoint
        End If
    End Function

    Private Function GetFillBlend() As Color()
        If Me.Enabled Then
            If _Checked Then
                If DrawState = eDrawState.Over Then
                    Return _OverColorBlendChecked
                ElseIf DrawState = eDrawState.Down Then
                    Return _ClickColorBlendChecked
                Else 'Up
                    Return _ColorFillBlendChecked.iColor
                End If
            Else
                If DrawState = eDrawState.Over Then
                    Return _OverColorBlend
                ElseIf DrawState = eDrawState.Down Then
                    Return _ClickColorBlend
                Else 'Up
                    Return _ColorFillBlend.iColor
                End If
            End If
        Else
            Return _DisabledBlend
        End If
    End Function

    Private Function GetFill() As Color
        If Me.Enabled Then
            If _Checked Then
                If DrawState = eDrawState.Over Then
                    Return _OverColorSolidChecked
                ElseIf DrawState = eDrawState.Down Then
                    Return _ClickColorSolidChecked
                Else 'Up
                    Return _ColorFillSolidChecked
                End If
            Else
                If DrawState = eDrawState.Over Then
                    Return _OverColorSolid
                ElseIf DrawState = eDrawState.Down Then
                    Return _ClickColorSolid
                Else 'Up
                    Return _ColorFillSolid
                End If
            End If
        Else
            Return _DisabledSolid
        End If
    End Function

    Function AdjustRect(ByVal BaseRect As RectangleF, ByVal Pad As Padding) As RectangleF
        BaseRect.Width -= Pad.Horizontal
        BaseRect.Height -= Pad.Vertical
        BaseRect.Offset(Pad.Left, Pad.Top)
        Return BaseRect
    End Function

    Function AdjustRect(ByVal BaseRect As Rectangle, ByVal Pad As Padding) As Rectangle
        BaseRect.Width -= Pad.Horizontal
        BaseRect.Height -= Pad.Vertical
        BaseRect.Offset(Pad.Left, Pad.Top)
        Return BaseRect
    End Function

    Public Function GetRoundedRectPath(ByVal BaseRect As RectangleF) As GraphicsPath
        Dim ArcRect As RectangleF
        Dim MyPath As New Drawing2D.GraphicsPath()
        If Me.Corners.All = -1 Then
            With MyPath
                ' top left arc
                If Me.Corners.UpperLeft = 0 Then
                    .AddLine(BaseRect.X, BaseRect.Y, BaseRect.X, BaseRect.Y)
                Else
                    ArcRect = New RectangleF(BaseRect.Location, _
                        New SizeF(Me.Corners.UpperLeft * 2, Me.Corners.UpperLeft * 2))
                    .AddArc(ArcRect, 180, 90)
                End If

                ' top right arc
                If Me.Corners.UpperRight = 0 Then
                    .AddLine(BaseRect.X + (Me.Corners.UpperLeft), BaseRect.Y, BaseRect.Right - (Me.Corners.UpperRight), BaseRect.Top)
                Else
                    ArcRect = New RectangleF(BaseRect.Location, _
                        New SizeF(Me.Corners.UpperRight * 2, Me.Corners.UpperRight * 2))
                    ArcRect.X = BaseRect.Right - (Me.Corners.UpperRight * 2)
                    .AddArc(ArcRect, 270, 90)
                End If

                ' bottom right arc
                If Me.Corners.LowerRight = 0 Then
                    .AddLine(BaseRect.Right, BaseRect.Top + (Me.Corners.UpperRight), BaseRect.Right, BaseRect.Bottom - (Me.Corners.LowerRight))
                Else
                    ArcRect = New RectangleF(BaseRect.Location, _
                        New SizeF(Me.Corners.LowerRight * 2, Me.Corners.LowerRight * 2))
                    ArcRect.Y = BaseRect.Bottom - (Me.Corners.LowerRight * 2)
                    ArcRect.X = BaseRect.Right - (Me.Corners.LowerRight * 2)
                    .AddArc(ArcRect, 0, 90)
                End If

                ' bottom left arc
                If Me.Corners.LowerLeft = 0 Then
                    .AddLine(BaseRect.Right - (Me.Corners.LowerRight), BaseRect.Bottom, BaseRect.X - (Me.Corners.LowerLeft), BaseRect.Bottom)
                Else
                    ArcRect = New RectangleF(BaseRect.Location, _
                        New SizeF(Me.Corners.LowerLeft * 2, Me.Corners.LowerLeft * 2))
                    ArcRect.Y = BaseRect.Bottom - (Me.Corners.LowerLeft * 2)
                    .AddArc(ArcRect, 90, 90)
                End If

                .CloseFigure()
            End With
        Else
            With MyPath
                If Me.Corners.All = 0 Then
                    .AddRectangle(BaseRect)
                Else

                    ArcRect = New RectangleF(BaseRect.Location, _
                        New SizeF(Me.Corners.All * 2, Me.Corners.All * 2))
                    ' top left arc
                    .AddArc(ArcRect, 180, 90)

                    ' top right arc
                    ArcRect.X = BaseRect.Right - (Me.Corners.All * 2)
                    .AddArc(ArcRect, 270, 90)

                    ' bottom right arc
                    ArcRect.Y = BaseRect.Bottom - (Me.Corners.All * 2)
                    .AddArc(ArcRect, 0, 90)

                    ' bottom left arc
                    ArcRect.X = BaseRect.Left
                    .AddArc(ArcRect, 90, 90)

                End If
                .CloseFigure()
            End With
        End If
        Return MyPath
    End Function

#End Region 'Drawing

#Region "MouseEvents"

    Private Sub MyButton_MouseDown(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles Me.MouseDown
        If DesignMode Then
            'Because of the GetHitTest Override in the
            'Designer Manual Selection is needed
            Dim selectservice As ISelectionService = _
                 CType(GetService(GetType(ISelectionService)), ISelectionService)
            Dim selection As New Collections.ArrayList
            selection.Clear()
            selectservice.SetSelectedComponents(selection, SelectionTypes.Replace)
            selection.Add(Me)
            selectservice.SetSelectedComponents(selection, SelectionTypes.Add)

            'FocusPoints Reset
            If e.Button = Windows.Forms.MouseButtons.Right Then
                If Me.CenterPtTracker.IsActive Then
                    If Checked Then
                        Me.FocalPointsChecked = New cFocalPoints(New PointF(0.5, 0.5), Me.FocalPointsChecked.FocusScales)
                    Else
                        Me.FocalPoints = New cFocalPoints(New PointF(0.5, 0.5), Me.FocalPoints.FocusScales)
                    End If
                    Me.Invalidate()
                ElseIf Me.FocusPtTracker.IsActive Then
                    If Checked Then
                        Me.FocalPointsChecked = New cFocalPoints(Me.FocalPointsChecked.CenterPoint, New PointF(0, 0))
                    Else
                        Me.FocalPoints = New cFocalPoints(Me.FocalPoints.CenterPoint, New PointF(0, 0))
                    End If
                    Me.Invalidate()
                End If
            End If
        Else
            If Me.Enabled Then
                If GetPath.IsVisible(e.X, e.Y) Then
                    DrawState = eDrawState.Down
                    Me.Invalidate(Rectangle.Round(ButtonArea))
                End If
            End If
        End If
    End Sub

    Private Sub MyButton_MouseMove(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles Me.MouseMove
        If DesignMode Then
            If e.Button = Windows.Forms.MouseButtons.Left Then
                If Me.CenterPtTracker.IsActive Then
                    If Checked Then
                        Me.FocalPointsChecked = New cFocalPoints(New PointF(CSng(e.X / Me.Width), CSng(e.Y / Me.Height)), _
                                                          Me.FocalPointsChecked.FocusScales)
                    Else
                        Me.FocalPoints = New cFocalPoints(New PointF(CSng(e.X / Me.Width), CSng(e.Y / Me.Height)), _
                                                          Me.FocalPoints.FocusScales)
                    End If
                    Me.Invalidate()
                ElseIf Me.FocusPtTracker.IsActive Then
                    If Checked Then
                        Me.FocalPointsChecked = New cFocalPoints(Me.FocalPointsChecked.CenterPoint, _
                                                          New PointF(CSng(e.X / Me.Width), CSng(e.Y / Me.Height)))
                    Else
                        Me.FocalPoints = New cFocalPoints(Me.FocalPoints.CenterPoint, _
                                                          New PointF(CSng(e.X / Me.Width), CSng(e.Y / Me.Height)))
                    End If
                    Me.Invalidate()
                End If
            End If
        Else
            If GetPath.IsVisible(e.X, e.Y) Then
                If Not DrawState = eDrawState.Down Then
                    DrawState = eDrawState.Over
                End If
                Me.Invalidate(Rectangle.Round(ButtonArea))
            Else
                If Not DrawState = eDrawState.Up Then
                    DrawState = eDrawState.Up
                    Me.Invalidate(Rectangle.Round(ButtonArea))
                End If
            End If
        End If
    End Sub

    Private Sub MyButton_MouseLeave(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.MouseLeave
        If Not DrawState = eDrawState.Up Then
            DrawState = eDrawState.Up
            Me.Invalidate(Rectangle.Round(ButtonArea))
        End If
    End Sub


    Public Event ClickButtonArea(ByVal Sender As Object, ByVal e As EventArgs)
    Public Event CheckedChanged(ByVal Sender As Object, ByVal e As EventArgs)

    Private Sub MyButton_MouseUp(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles Me.MouseUp
        If _CheckButton And Not DesignMode Then
            _Checked = Not _Checked
            RaiseEvent CheckedChanged(Me, New EventArgs)
        End If
        If DrawState = eDrawState.Down Then RaiseEvent ClickButtonArea(Me, New EventArgs)
        DrawState = eDrawState.Up
        Me.Invalidate(Rectangle.Round(ButtonArea))
    End Sub

#End Region 'MouseEvents

#Region "Control Events"

    Private Sub MyButton_Resize(ByVal sender As Object, ByVal e As System.EventArgs) Handles Me.Resize
        If DesignMode Then
            CenterPtTracker.TrackerRectangle = CenterPtTrackerRectangle()
            FocusPtTracker.TrackerRectangle = FocusPtTrackerRectangle()
        End If
    End Sub

#End Region 'Control Events

End Class

#End Region 'MyButton Class

#Region "DesignerRectTracker"

Public Class DesignerRectTracker

    Private _TrackerRectangle As RectangleF
    Public Property TrackerRectangle() As RectangleF
        Get
            Return _TrackerRectangle
        End Get
        Set(ByVal value As RectangleF)
            _TrackerRectangle = value
        End Set
    End Property

    Private _IsActive As Boolean
    Public Property IsActive() As Boolean
        Get
            Return _IsActive
        End Get
        Set(ByVal value As Boolean)
            _IsActive = value
        End Set
    End Property

    Sub New()
        _TrackerRectangle = New RectangleF(0, 0, 10, 10)
        _IsActive = False
    End Sub

End Class

#End Region 'DesignerRectTracker

#Region "BlendTypeEditor - UITypeEditor"

#Region "cBlendItems"

Public Class cBlendItems

    Sub New()

    End Sub

    Sub New(ByVal Color As Color(), ByVal Pt As Single())
        iColor = Color
        iPoint = Pt
    End Sub

    Private _iColor As Color()
    <Description("The Color for the Point"), _
       Category("Appearance")> _
    Public Property iColor() As Color()
        Get
            Return _iColor
        End Get
        Set(ByVal value As Color())
            _iColor = value
        End Set
    End Property

    Private _iPoint As Single()
    <Description("The Color for the Point"), _
       Category("Appearance")> _
    Public Property iPoint() As Single()
        Get
            Return _iPoint
        End Get
        Set(ByVal value As Single())
            _iPoint = value
        End Set
    End Property

    Public Overrides Function ToString() As String
        Return "BlendItems"
    End Function

End Class

#End Region 'cBlendItems

Public Class BlendTypeEditor
    Inherits UITypeEditor

    Public Overloads Overrides Function GetEditStyle(ByVal context As ITypeDescriptorContext) As UITypeEditorEditStyle
        If Not context Is Nothing Then
            Return UITypeEditorEditStyle.DropDown
        End If
        Return (MyBase.GetEditStyle(context))
    End Function

    Public Overloads Overrides Function EditValue(ByVal context As ITypeDescriptorContext, ByVal provider As IServiceProvider, ByVal value As Object) As Object
        If (Not context Is Nothing) And (Not provider Is Nothing) Then
            ' Access the property browser's UI display service, IWindowsFormsEditorService
            Dim editorService As IWindowsFormsEditorService = CType(provider.GetService(GetType(IWindowsFormsEditorService)), IWindowsFormsEditorService)
            If Not editorService Is Nothing Then
                ' Create an instance of the UI editor, passing a reference to the editor service
                Dim dropDownEditor As DropdownColorBlender = New DropdownColorBlender(editorService, context)

                ' Pass the UI editor the current property values
                Dim Instance As MyButton
                If context.Instance.GetType Is GetType(MyButton) Then
                    'For PropertyGrid
                    Instance = CType(context.Instance, MyButton)
                Else
                    'For SmartTag
                    Instance = CType(CType(context.Instance, MyButtonActionList).CurrControl, MyButton)
                End If

                'Update The Sample with the Current Instance's Properties
                With dropDownEditor
                    Dim ratio As Single
                    If Instance.Width > Instance.Height Then
                        .TheSample.Height = CInt(.TheSample.Width * (Instance.Height / Instance.Width))
                        .TheSample.Top = CInt((.panSampleHolder.Height - .TheSample.Height) / 2)
                        ratio = CSng(.TheSample.Height / Instance.Height)
                    Else
                        .TheSample.Width = CInt(.TheSample.Height * (Instance.Width / Instance.Height))
                        .TheSample.Left = CInt((.panSampleHolder.Width - .TheSample.Width) / 2)
                        ratio = CSng(.TheSample.Width / Instance.Width)
                    End If

                    ' Set current Corners values
                    .TheSample.Shape = Instance.Shape
                    .TheSample.FillType = Instance.FillType
                    .TheSample.FillTypeChecked = Instance.FillTypeChecked
                    '
                    .TheSample.CheckButton = Instance.CheckButton

                    If context.PropertyDescriptor.DisplayName = "ColorFillBlend" Then
                        .TheSample.Checked = False
                    Else
                        .TheSample.Checked = True
                    End If

                    .TheSample.ColorFillSolid = Instance.ColorFillSolid
                    .TheSample.ColorFillBlend = Instance.ColorFillBlend
                    .TheSample.ColorFillSolidChecked = Instance.ColorFillSolidChecked
                    .TheSample.ColorFillBlendChecked = Instance.ColorFillBlendChecked
                    '
                    .TheSample.BorderColor = Instance.BorderColor
                    '
                    .TheSample.FocalPoints = Instance.FocalPoints
                    .TheSample.FocalPointsChecked = Instance.FocalPointsChecked

                    .TheSample.Corners = New CornersProperty( _
                                    CShort(Instance.Corners.LowerLeft * ratio), _
                                    CShort(Instance.Corners.LowerRight * ratio), _
                                    CShort(Instance.Corners.UpperLeft * ratio), _
                                    CShort(Instance.Corners.UpperRight * ratio))

                    ' -------------------------------------------------- use the appropriate property value
                    If context.PropertyDescriptor.DisplayName = "ColorFillBlend" Then
                        .LoadABlend(Instance.ColorFillBlend)
                    Else
                        .LoadABlend(Instance.ColorFillBlendChecked)
                    End If

                    .TheSample.TextMargin = New Padding( _
                                    CInt(Instance.TextMargin.Left * ratio), _
                                    CInt(Instance.TextMargin.Top * ratio), _
                                    CInt(Instance.TextMargin.Right * ratio), _
                                    CInt(Instance.TextMargin.Bottom * ratio))
                    .TheSample.Padding = New Padding( _
                                    CInt(Instance.Padding.Left * ratio), _
                                    CInt(Instance.Padding.Top * ratio), _
                                    CInt(Instance.Padding.Right * ratio), _
                                    CInt(Instance.Padding.Bottom * ratio))
                    .TheSample.Text = Instance.Text
                    .TheSample.ForeColor = Instance.ForeColor
                    .TheSample.ForeColorChecked = Instance.ForeColorChecked
                    .TheSample.TextAlign = Instance.TextAlign
                    .TheSample.Font = New Font(Instance.Font.FontFamily, Instance.Font.Size * ratio, Instance.Font.Style)
                    .TheSample.TextShadow = Instance.TextShadow
                    .TheSample.TextShadowChecked = Instance.TextShadowChecked
                End With

                ' -------------------------------------------------- Display the UI editor
                editorService.DropDownControl(dropDownEditor)

                ' -------------------------------------------------- Return the new property value from the editor
                'MsgBox(context.PropertyDescriptor.DisplayName)
                If context.PropertyDescriptor.DisplayName = "ColorFillBlend" Then
                    Return dropDownEditor.TheSample.ColorFillBlend
                Else
                    Return dropDownEditor.TheSample.ColorFillBlendChecked
                End If

            End If
        End If
        Return MyBase.EditValue(context, provider, value)
    End Function

    ' Indicate that we draw values in the Properties window.
    Public Overrides Function GetPaintValueSupported(ByVal context As System.ComponentModel.ITypeDescriptorContext) As Boolean
        Return True
    End Function

    ' Draw a BorderStyles value.
    Public Overrides Sub PaintValue(ByVal e As System.Drawing.Design.PaintValueEventArgs)
        ' Erase the area.
        e.Graphics.FillRectangle(Brushes.White, e.Bounds)

        ' Draw the sample.
        Dim cblnd As cBlendItems = DirectCast(e.Value, cBlendItems)
        Using br As LinearGradientBrush = New LinearGradientBrush(e.Bounds, Color.Black, Color.Black, LinearGradientMode.Horizontal)
            Dim cb As New ColorBlend
            cb.Colors = cblnd.iColor
            cb.Positions = cblnd.iPoint
            br.InterpolationColors = cb
            e.Graphics.FillRectangle(br, e.Bounds)
        End Using
    End Sub
End Class
#End Region 'BlendTypeEditor - UITypeEditor

#Region "Expandable Focal Points Property Classes"

#Region "cFocalPoints"

<TypeConverter(GetType(FocalPointsConverter))> _
Public Class cFocalPoints

    Private _CenterPtX As Single = 0.5
    <RefreshProperties(RefreshProperties.Repaint)> _
    <NotifyParentProperty(True)> _
    <DefaultValue(0.5)> _
        Public Property CenterPtX() As Single
        Get
            Return _CenterPtX
        End Get
        Set(ByVal value As Single)
            If value < 0 Then value = 0
            If value > 1 Then value = 1
            _CenterPtX = value
        End Set
    End Property

    Private _CenterPtY As Single = 0.5
    <RefreshProperties(RefreshProperties.Repaint)> _
    <NotifyParentProperty(True)> _
    <DefaultValue(0.5)> _
    Public Property CenterPtY() As Single
        Get
            Return _CenterPtY
        End Get
        Set(ByVal value As Single)
            If value < 0 Then value = 0
            If value > 1 Then value = 1
            _CenterPtY = value
        End Set
    End Property

    Private _FocusPtX As Single = 0
    <RefreshProperties(RefreshProperties.Repaint)> _
    <NotifyParentProperty(True)> _
    <DefaultValue(0)> _
    Public Property FocusPtX() As Single
        Get
            Return _FocusPtX
        End Get
        Set(ByVal value As Single)
            If value < 0 Then value = 0
            If value > 1 Then value = 1
            _FocusPtX = value
        End Set
    End Property

    Private _FocusPtY As Single = 0
    <RefreshProperties(RefreshProperties.Repaint)> _
    <NotifyParentProperty(True)> _
    <DefaultValue(0)> _
    Public Property FocusPtY() As Single
        Get
            Return _FocusPtY
        End Get
        Set(ByVal value As Single)
            If value < 0 Then value = 0
            If value > 1 Then value = 1
            _FocusPtY = value
        End Set
    End Property

    Public Function CenterPoint() As PointF
        Return New PointF(Me.CenterPtX, Me.CenterPtY)
    End Function

    Public Function FocusScales() As PointF
        Return New PointF(Me.FocusPtX, Me.FocusPtY)
    End Function

    Sub New()
        Me.CenterPtX = 0.5
        Me.CenterPtY = 0.5
        Me.FocusPtX = 0
        Me.FocusPtY = 0
    End Sub

    Sub New(ByVal Cx As Single, ByVal Cy As Single, ByVal Fx As Single, ByVal Fy As Single)
        Me.CenterPtX = Cx
        Me.CenterPtY = Cy
        Me.FocusPtX = Fx
        Me.FocusPtY = Fy
    End Sub

    Sub New(ByVal ptC As PointF, ByVal ptF As PointF)
        Me.CenterPtX = ptC.X
        Me.CenterPtY = ptC.Y
        Me.FocusPtX = ptF.X
        Me.FocusPtY = ptF.Y
    End Sub

    Public Overrides Function ToString() As String
        Return String.Format("{0}, {1}, {2}, {3}", _CenterPtX, _CenterPtY, _FocusPtX, _FocusPtY)
    End Function

End Class

#End Region 'cFocalPoints

#Region "FocalPointsConverter"

Friend Class FocalPointsConverter : Inherits ExpandableObjectConverter

    Public Overrides Function GetCreateInstanceSupported(ByVal context As System.ComponentModel.ITypeDescriptorContext) As Boolean
        Return True
    End Function

    Public Overrides Function CreateInstance(ByVal context As System.ComponentModel.ITypeDescriptorContext, ByVal propertyValues As System.Collections.IDictionary) As Object
        Dim fPt As New cFocalPoints
        fPt.CenterPtX = CType(propertyValues("CenterPtX"), Single)
        fPt.CenterPtY = CType(propertyValues("CenterPtY"), Single)
        fPt.FocusPtX = CType(propertyValues("FocusPtX"), Single)
        fPt.FocusPtY = CType(propertyValues("FocusPtY"), Single)
        Return fPt
    End Function

    Public Overloads Overrides Function CanConvertFrom(ByVal context As System.ComponentModel.ITypeDescriptorContext, ByVal sourceType As System.Type) As Boolean
        If (sourceType Is GetType(String)) Then
            Return True
        End If
        Return MyBase.CanConvertFrom(context, sourceType)
    End Function

    Public Overloads Overrides Function ConvertFrom(ByVal context As System.ComponentModel.ITypeDescriptorContext, _
      ByVal culture As System.Globalization.CultureInfo, ByVal value As Object) As Object
        If TypeOf value Is String Then
            Try
                Dim s As String = CType(value, String)
                Dim FocalPointsParts(4) As String
                FocalPointsParts = Split(s, ",")
                If Not IsNothing(FocalPointsParts) Then
                    If IsNothing(FocalPointsParts(0)) Then FocalPointsParts(0) = "0.5"
                    If IsNothing(FocalPointsParts(1)) Then FocalPointsParts(1) = "0.5"
                    If IsNothing(FocalPointsParts(2)) Then FocalPointsParts(2) = "0"
                    If IsNothing(FocalPointsParts(3)) Then FocalPointsParts(3) = "0"
                    Return New cFocalPoints(CSng(FocalPointsParts(0).Trim), _
                                            CSng(FocalPointsParts(1).Trim), _
                                            CSng(FocalPointsParts(2).Trim), _
                                            CSng(FocalPointsParts(3).Trim))
                End If
            Catch ex As Exception
                Throw New ArgumentException("Can not convert '" & CStr(value) & "' to type Corners")
            End Try
        Else
            Return New CornersProperty()
        End If

        Return MyBase.ConvertFrom(context, culture, value)
    End Function

    Public Overloads Overrides Function ConvertTo(ByVal context As System.ComponentModel.ITypeDescriptorContext, _
      ByVal culture As System.Globalization.CultureInfo, _
      ByVal value As Object, ByVal destinationType As System.Type) As Object

        If (destinationType Is GetType(System.String) AndAlso TypeOf value Is cFocalPoints) Then
            Dim _FocalPoints As cFocalPoints = CType(value, cFocalPoints)

            ' build the string as "UpperLeft,UpperRight,LowerLeft,LowerRight" 
            Return String.Format("{0}, {1}, {2}, {3}", _FocalPoints.CenterPtX, _FocalPoints.CenterPtY, _FocalPoints.FocusPtX, _FocalPoints.FocusPtY)
        End If
        Return MyBase.ConvertTo(context, culture, value, destinationType)

    End Function

End Class 'CornerConverter Code

#End Region 'FocalPointsConverter

#End Region 'Expandable Focal Points Property Class

#Region "Expandable Border Corners Property Class"

#Region "CornersProperty"

<TypeConverter(GetType(CornerConverter))> _
Public Class CornersProperty

    Private _All As Short = -1
    Private _UpperLeft As Short = 0
    Private _UpperRight As Short = 0
    Private _LowerLeft As Short = 0
    Private _LowerRight As Short = 0

    Public Sub New(ByVal LowerLeft As Short, ByVal LowerRight As Short, _
      ByVal UpperLeft As Short, ByVal UpperRight As Short)
        Me.LowerLeft = LowerLeft
        Me.LowerRight = LowerRight
        Me.UpperLeft = UpperLeft
        Me.UpperRight = UpperRight
    End Sub

    Public Sub New(ByVal All As Short)
        Me.All = All
    End Sub

    Public Sub New()
        Me.LowerLeft = 0
        Me.LowerRight = 0
        Me.UpperLeft = 0
        Me.UpperRight = 0
    End Sub

    Private Sub CheckForAll(ByVal val As Short)
        If val = LowerLeft AndAlso _
           val = LowerRight AndAlso _
           val = UpperLeft AndAlso _
           val = UpperRight Then
            If _All <> val Then _All = val
        Else
            If All <> -1 Then All = -1
        End If
    End Sub

    <DescriptionAttribute("Set the Radius of the All four Corners the same")> _
    <RefreshProperties(RefreshProperties.Repaint)> _
    <NotifyParentProperty(True)> _
    <DefaultValue(-1)> _
    Public Property All() As Short
        Get
            Return _All
        End Get
        Set(ByVal Value As Short)
            _All = Value
            If Value > -1 Then
                Me._LowerLeft = Value
                Me._LowerRight = Value
                Me._UpperLeft = Value
                Me._UpperRight = Value
            End If
        End Set

    End Property

    <DescriptionAttribute("Set the Radius of the Upper Left Corner")> _
    <RefreshProperties(RefreshProperties.Repaint)> _
    <NotifyParentProperty(True)> _
    <DefaultValue(0)> _
    Public Property UpperLeft() As Short
        Get
            Return _UpperLeft
        End Get
        Set(ByVal Value As Short)
            _UpperLeft = Value

            CheckForAll(Value)
        End Set
    End Property

    <DescriptionAttribute("Set the Radius of the Upper Right Corner")> _
    <RefreshProperties(RefreshProperties.Repaint)> _
    <NotifyParentProperty(True)> _
    <DefaultValue(0)> _
    Public Property UpperRight() As Short
        Get
            Return _UpperRight
        End Get
        Set(ByVal Value As Short)
            _UpperRight = Value
            CheckForAll(Value)
        End Set
    End Property

    <DescriptionAttribute("Set the Radius of the Lower Left Corner")> _
    <RefreshProperties(RefreshProperties.Repaint)> _
    <NotifyParentProperty(True)> _
    <DefaultValue(0)> _
    Public Property LowerLeft() As Short
        Get
            Return _LowerLeft
        End Get
        Set(ByVal Value As Short)
            _LowerLeft = Value
            CheckForAll(Value)
        End Set
    End Property

    <DescriptionAttribute("Set the Radius of the Lower Right Corner")> _
    <RefreshProperties(RefreshProperties.Repaint)> _
    <NotifyParentProperty(True)> _
    <DefaultValue(0)> _
    Public Property LowerRight() As Short
        Get
            Return _LowerRight
        End Get
        Set(ByVal Value As Short)
            _LowerRight = Value
            CheckForAll(Value)
        End Set
    End Property

End Class 'Corners Properties

#End Region 'CornersProperty

#Region "CornerConverter"

Friend Class CornerConverter : Inherits ExpandableObjectConverter

    Public Overrides Function GetCreateInstanceSupported(ByVal context As System.ComponentModel.ITypeDescriptorContext) As Boolean
        Return True
    End Function

    Public Overrides Function CreateInstance(ByVal context As System.ComponentModel.ITypeDescriptorContext, ByVal propertyValues As System.Collections.IDictionary) As Object
        Dim crn As New CornersProperty
        Dim AL As Short = CType(propertyValues("All"), Short)
        Dim LL As Short = CType(propertyValues("LowerLeft"), Short)
        Dim LR As Short = CType(propertyValues("LowerRight"), Short)
        Dim UL As Short = CType(propertyValues("UpperLeft"), Short)
        Dim UR As Short = CType(propertyValues("UpperRight"), Short)


        Dim oAll As Short = CType(CType(context.Instance, MyButton).Corners, CornersProperty).All

        If oAll <> AL And AL > -1 Then
            crn.All = AL
        Else
            crn.LowerLeft = LL
            crn.LowerRight = LR
            crn.UpperLeft = UL
            crn.UpperRight = UR

        End If

        Return crn
    End Function

    Public Overloads Overrides Function CanConvertFrom(ByVal context As System.ComponentModel.ITypeDescriptorContext, _
      ByVal sourceType As System.Type) As Boolean

        If (sourceType Is GetType(String)) Then
            Return True
        End If
        Return MyBase.CanConvertFrom(context, sourceType)
    End Function

    Public Overloads Overrides Function ConvertFrom(ByVal context As System.ComponentModel.ITypeDescriptorContext, _
      ByVal culture As System.Globalization.CultureInfo, ByVal value As Object) As Object
        If TypeOf value Is String Then
            Try
                Dim s As String = CType(value, String)
                Dim cornerParts(4) As String
                cornerParts = Split(s, ",")
                If Not IsNothing(cornerParts) Then
                    If IsNothing(cornerParts(0)) Then cornerParts(0) = "0"
                    If IsNothing(cornerParts(1)) Then cornerParts(1) = "0"
                    If IsNothing(cornerParts(2)) Then cornerParts(2) = "0"
                    If IsNothing(cornerParts(3)) Then cornerParts(3) = "0"
                    Return New CornersProperty(CShort(cornerParts(0).Trim), _
                                               CShort(cornerParts(1).Trim), _
                                               CShort(cornerParts(2).Trim), _
                                               CShort(cornerParts(3).Trim))
                End If
            Catch ex As Exception
                Throw New ArgumentException("Can not convert '" & CStr(value) & "' to type Corners")
            End Try
        Else
            Return New CornersProperty()
        End If

        Return MyBase.ConvertFrom(context, culture, value)
    End Function

    Public Overloads Overrides Function ConvertTo(ByVal context As System.ComponentModel.ITypeDescriptorContext, _
      ByVal culture As System.Globalization.CultureInfo, _
      ByVal value As Object, ByVal destinationType As System.Type) As Object

        Dim _Corners As CornersProperty = CType(value, CornersProperty)
        If (destinationType Is GetType(System.String) AndAlso TypeOf value Is CornersProperty) Then
            ' build the string as "UpperLeft, UpperRight, LowerLeft, LowerRight" 
            Return String.Format("{0}, {1}, {2}, {3}", _
                _Corners.LowerLeft, _
                _Corners.LowerRight, _
                _Corners.UpperLeft, _
                _Corners.UpperRight)
        Else
            Return MyBase.ConvertTo(context, culture, value, destinationType)
        End If

    End Function

End Class 'CornerConverter Code

#End Region 'CornerConverter

#End Region 'Expandable Border Corners Property Class

