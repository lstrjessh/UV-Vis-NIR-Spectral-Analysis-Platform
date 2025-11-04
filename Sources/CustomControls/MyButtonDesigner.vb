
Imports System.Windows.Forms.Design
Imports System.ComponentModel
Imports System.ComponentModel.Design
Imports System.Drawing.Design
Imports System.Drawing.Drawing2D
Imports System.Drawing
Imports System.Windows.Forms

#Region "MyButtonDesigner"

Public Class MyButtonDesigner
    Inherits ControlDesigner

    Private _MyButton As MyButton = Nothing
    Private _Lists As DesignerActionListCollection
    Private InABox As Boolean = False

    Public Overrides Sub Initialize(ByVal component As IComponent)
        MyBase.Initialize(component)

        ' Get clock control shortcut reference
        _MyButton = CType(component, MyButton)
    End Sub

    Protected Overrides Function GetHitTest(ByVal pt As System.Drawing.Point) As Boolean
        pt = _MyButton.PointToClient(pt)
        _MyButton.CenterPtTracker.IsActive = _
            _MyButton.CenterPtTracker.TrackerRectangle.Contains(pt)
        _MyButton.FocusPtTracker.IsActive = _
            _MyButton.FocusPtTracker.TrackerRectangle.Contains(pt)
        Return _MyButton.CenterPtTracker.IsActive Or _MyButton.FocusPtTracker.IsActive
    End Function

    Protected Overrides Sub OnMouseEnter()
        MyBase.OnMouseEnter()
        InABox = True
        _MyButton.Invalidate()
    End Sub

    Protected Overrides Sub OnMouseLeave()
        MyBase.OnMouseLeave()
        InABox = False
        _MyButton.Invalidate()
    End Sub

    Protected Overrides Sub OnPaintAdornments _
      (ByVal pe As System.Windows.Forms.PaintEventArgs)
        MyBase.OnPaintAdornments(pe)
        '
        Dim ft As MyButton.eFillType
        If _MyButton.Checked Then
            ft = _MyButton.FillTypeChecked
        Else
            ft = _MyButton.FillType
        End If
        '
        If ft = MyButton.eFillType.GradientPath And InABox Then
            Using g As Graphics = pe.Graphics
                Using pn As Pen = New Pen(Color.Gray, 1)
                    pn.DashStyle = DashStyle.Dot
                    g.FillEllipse( _
                        New SolidBrush(Color.FromArgb(100, 255, 255, 255)), _
                        _MyButton.CenterPtTracker.TrackerRectangle)
                    g.DrawEllipse(pn, _MyButton.CenterPtTracker.TrackerRectangle)

                    g.FillRectangle( _
                        New SolidBrush(Color.FromArgb(100, 255, 255, 255)), _
                        _MyButton.FocusPtTracker.TrackerRectangle)
                    Dim rect As RectangleF = _MyButton.FocusPtTracker.TrackerRectangle
                    g.DrawRectangle(pn, rect.X, rect.Y, rect.Width, rect.Height)
                End Using
            End Using
        End If
    End Sub

#Region "ActionLists"

    Public Overrides ReadOnly Property ActionLists() As System.ComponentModel.Design.DesignerActionListCollection
        Get
            If _Lists Is Nothing Then
                _Lists = New DesignerActionListCollection
                _Lists.Add(New MyButtonActionList(Me.Component))
            End If
            Return _Lists
        End Get
    End Property

#End Region 'ActionLists

End Class

#End Region 'MyButtonDesigner

#Region "MyButtonActionList"

Public Class MyButtonActionList
    Inherits DesignerActionList

    Private _MyButton As MyButton
    Private _DesignerService As DesignerActionUIService = Nothing

    Public Sub New(ByVal component As IComponent)
        MyBase.New(component)

        ' Save a reference to the control we are designing.
        _MyButton = DirectCast(component, MyButton)

        ' Save a reference to the DesignerActionUIService
        _DesignerService = _
            CType(GetService(GetType(DesignerActionUIService)),  _
            DesignerActionUIService)

        'Makes the Smart Tags open automatically 
        Me.AutoShow = True
    End Sub

#Region "Smart Tag Items"

#Region "Properties"

    Public ReadOnly Property CurrControl() As MyButton
        Get
            Return _MyButton
        End Get
    End Property

#Region "Shape"

    Public Property Shape() As MyButton.eShape
        Get
            Return _MyButton.Shape
        End Get
        Set(ByVal value As MyButton.eShape)
            SetControlProperty("Shape", value)
        End Set
    End Property

#End Region 'Shape

#Region "Border"

    Public Property BorderColor() As Color
        Get
            Return _MyButton.BorderColor
        End Get
        Set(ByVal value As Color)
            SetControlProperty("BorderColor", value)
        End Set
    End Property

    Public Property BorderShow() As Boolean
        Get
            Return _MyButton.BorderShow
        End Get
        Set(ByVal value As Boolean)
            SetControlProperty("BorderShow", value)
        End Set
    End Property

#End Region 'Border

#Region "Fill"

    <Editor(GetType(BlendTypeEditor), GetType(UITypeEditor))> _
    Public Property ColorFillBlend() As cBlendItems
        Get
            Return _MyButton.ColorFillBlend
        End Get
        Set(ByVal value As cBlendItems)
            SetControlProperty("ColorFillBlend", value)
        End Set
    End Property

    <Editor(GetType(BlendTypeEditor), GetType(UITypeEditor))> _
    Public Property ColorFillBlendChecked() As cBlendItems
        Get
            Return _MyButton.ColorFillBlendChecked
        End Get
        Set(ByVal value As cBlendItems)
            SetControlProperty("ColorFillBlendChecked", value)
        End Set
    End Property

    Public Property FillType() As MyButton.eFillType
        Get
            Return _MyButton.FillType
        End Get
        Set(ByVal value As MyButton.eFillType)
            SetControlProperty("FillType", value)
        End Set
    End Property

    Public Property FillTypeChecked() As MyButton.eFillType
        Get
            Return _MyButton.FillTypeChecked
        End Get
        Set(ByVal value As MyButton.eFillType)
            SetControlProperty("FillTypeChecked", value)
        End Set
    End Property

    Public Property ColorFillSolid() As Color
        Get
            Return _MyButton.ColorFillSolid
        End Get
        Set(ByVal value As Color)
            SetControlProperty("ColorFillSolid", value)
        End Set
    End Property

    Public Property ColorFillSolidChecked() As Color
        Get
            Return _MyButton.ColorFillSolidChecked
        End Get
        Set(ByVal value As Color)
            SetControlProperty("ColorFillSolidChecked", value)
        End Set
    End Property

#End Region 'Fill

#Region "Text"

    Public Property Text() As String
        Get
            Return _MyButton.Text
        End Get
        Set(ByVal value As String)
            SetControlProperty("Text", value)
        End Set
    End Property

    Public Property TextAlign() As ContentAlignment
        Get
            Return _MyButton.TextAlign
        End Get
        Set(ByVal value As ContentAlignment)
            SetControlProperty("TextAlign", value)
        End Set
    End Property

    Public Property TextImageRelation() As TextImageRelation
        Get
            Return _MyButton.TextImageRelation
        End Get
        Set(ByVal value As TextImageRelation)
            SetControlProperty("TextImageRelation", value)
        End Set
    End Property

    Public Property ForeColor() As Color
        Get
            Return _MyButton.ForeColor
        End Get
        Set(ByVal value As Color)
            SetControlProperty("ForeColor", value)
        End Set
    End Property

    Public Property ForeColorChecked() As Color
        Get
            Return _MyButton.ForeColorChecked
        End Get
        Set(ByVal value As Color)
            SetControlProperty("ForeColorChecked", value)
        End Set
    End Property

    Public Property TextShadow() As Color
        Get
            Return _MyButton.TextShadow
        End Get
        Set(ByVal value As Color)
            SetControlProperty("TextShadow", value)
        End Set
    End Property

    Public Property TextShadowChecked() As Color
        Get
            Return _MyButton.TextShadowChecked
        End Get
        Set(ByVal value As Color)
            SetControlProperty("TextShadowChecked", value)
        End Set
    End Property

#End Region 'Text

#Region "Image"

    Public Property ImageAlign() As ContentAlignment
        Get
            Return _MyButton.ImageAlign
        End Get
        Set(ByVal value As ContentAlignment)
            SetControlProperty("ImageAlign", value)
        End Set
    End Property

    Public Property SideImageAlign() As ContentAlignment
        Get
            Return _MyButton.SideImageAlign
        End Get
        Set(ByVal value As ContentAlignment)
            SetControlProperty("SideImageAlign", value)
        End Set
    End Property

    Public Property SideImageBehindText() As Boolean
        Get
            Return _MyButton.SideImageBehindText
        End Get
        Set(ByVal value As Boolean)
            SetControlProperty("SideImageBehindText", value)
        End Set
    End Property

#End Region '

#End Region 'Properties

#Region "Methods"

    Public Sub AdjustCorners()

        'Create a new Corners Dialog and update the controls on the form
        Dim dlg As dlgCorners = New dlgCorners()

        Dim maxcorner As Integer
        Dim ratio As Single
        If _MyButton.Width > _MyButton.Height Then
            dlg.TheSample.Height = CInt(dlg.TheSample.Width * (_MyButton.Height / _MyButton.Width))
            dlg.TheSample.Top = CInt((dlg.panSampleHolder.Height - dlg.TheSample.Height) / 2)
            ratio = CSng(dlg.TheSample.Height / _MyButton.Height)
            maxcorner = CInt(((dlg.TheSample.Height / 2)) - (((_MyButton.Padding.Top * ratio) + (_MyButton.Padding.Bottom * ratio)) / 2))
        Else
            dlg.TheSample.Width = CInt(dlg.TheSample.Height * (_MyButton.Width / _MyButton.Height))
            dlg.TheSample.Left = CInt((dlg.panSampleHolder.Width - dlg.TheSample.Width) / 2)
            maxcorner = CInt(((dlg.TheSample.Width / 2)) - ((_MyButton.Padding.Left + _MyButton.Padding.Right) / 2))
            ratio = CSng(dlg.TheSample.Width / _MyButton.Width)
        End If

        ' Set current Corners values
        With dlg
            .tbarUpperLeft.Maximum = maxcorner
            .tbarUpperRight.Maximum = maxcorner
            .tbarLowerLeft.Maximum = maxcorner
            .tbarLowerRight.Maximum = maxcorner
            .tbarAll.Maximum = maxcorner
            .tbarUpperLeft.TickFrequency = CInt(maxcorner / 2)
            .tbarUpperRight.TickFrequency = CInt(maxcorner / 2)
            .tbarLowerLeft.TickFrequency = CInt(maxcorner / 2)
            .tbarLowerRight.TickFrequency = CInt(maxcorner / 2)
            .tbarAll.TickFrequency = CInt(maxcorner / 2)
            If _MyButton.Corners.All > -1 Then
                .tbarAll.Value = CInt(Math.Min((_MyButton.Corners.UpperLeft * ratio), maxcorner))
            End If
            .tbarUpperLeft.Value = CInt(Math.Min((_MyButton.Corners.UpperLeft * ratio), maxcorner))
            .tbarUpperRight.Value = CInt(Math.Min((_MyButton.Corners.UpperRight * ratio), maxcorner))
            .tbarLowerLeft.Value = CInt(Math.Min((_MyButton.Corners.LowerLeft * ratio), maxcorner))
            .tbarLowerRight.Value = CInt(Math.Min((_MyButton.Corners.LowerRight * ratio), maxcorner))

            .TheSample.Shape = _MyButton.Shape
            .TheSample.FillType = _MyButton.FillType
            .TheSample.FillTypeChecked = _MyButton.FillTypeChecked

            .TheSample.ColorFillSolid = _MyButton.ColorFillSolid
            .TheSample.BorderColor = _MyButton.BorderColor
            .TheSample.ColorFillBlend = _MyButton.ColorFillBlend

            .TheSample.CheckButton = _MyButton.CheckButton
            .TheSample.Checked = _MyButton.Checked
            .TheSample.ColorFillBlend = _MyButton.ColorFillBlendChecked
            .TheSample.ColorFillSolid = _MyButton.ColorFillSolidChecked

            .TheSample.Corners = New CornersProperty( _
                            CShort(_MyButton.Corners.LowerLeft * ratio), _
                            CShort(_MyButton.Corners.LowerRight * ratio), _
                            CShort(_MyButton.Corners.UpperLeft * ratio), _
                            CShort(_MyButton.Corners.UpperRight * ratio))
            .TheSample.FocalPoints = _MyButton.FocalPoints
            .TheSample.FocalPointsChecked = _MyButton.FocalPointsChecked
            .TheSample.TextMargin = New Padding( _
                            CInt(_MyButton.TextMargin.Left * ratio), _
                            CInt(_MyButton.TextMargin.Top * ratio), _
                            CInt(_MyButton.TextMargin.Right * ratio), _
                            CInt(_MyButton.TextMargin.Bottom * ratio))
            .TheSample.Padding = New Padding( _
                            CInt(_MyButton.Padding.Left * ratio), _
                            CInt(_MyButton.Padding.Top * ratio), _
                            CInt(_MyButton.Padding.Right * ratio), _
                            CInt(_MyButton.Padding.Bottom * ratio))
            .TheSample.Text = _MyButton.Text
            .TheSample.ForeColor = _MyButton.ForeColor
            .TheSample.TextAlign = _MyButton.TextAlign
            .TheSample.Font = New Font(_MyButton.Font.FontFamily, _MyButton.Font.Size * ratio, _MyButton.Font.Style)
            .TheSample.TextShadow = _MyButton.TextShadow

            .ratio = ratio

        End With


        ' Update new Corners values if OK button was pressed
        If dlg.ShowDialog() = DialogResult.OK Then
            Dim designerHost As IDesignerHost = CType(Me.Component.Site.GetService(GetType(IDesignerHost)), IDesignerHost)

            If designerHost IsNot Nothing Then
                Dim t As DesignerTransaction = designerHost.CreateTransaction()
                Try
                    SetControlProperty("Corners", New CornersProperty( _
                        CShort(dlg.TheSample.Corners.LowerLeft / ratio), _
                        CShort(dlg.TheSample.Corners.LowerRight / ratio), _
                        CShort(dlg.TheSample.Corners.UpperLeft / ratio), _
                        CShort(dlg.TheSample.Corners.UpperRight / ratio)))
                    t.Commit()
                Catch
                    t.Cancel()
                End Try
            End If
        End If
        _MyButton.Refresh()
        _DesignerService.Refresh(_MyButton)

    End Sub

#End Region 'Methods

    ' Set a control property. This method makes Undo/Redo
    ' work properly and marks the form as modified in the IDE.
    Private Sub SetControlProperty(ByVal property_name As String, ByVal value As Object)
        TypeDescriptor.GetProperties(_MyButton) _
            (property_name).SetValue(_MyButton, value)
    End Sub

#End Region ' Smart Tag Items

    ' Return the smart tag action items.
    Public Overrides Function GetSortedActionItems() As System.ComponentModel.Design.DesignerActionItemCollection
        Dim items As New DesignerActionItemCollection()

        items.Add( _
            New DesignerActionPropertyItem( _
                "Shape", _
                "Shape", _
                "", _
                "The Shape of the Control"))

        items.Add(New DesignerActionHeaderItem("Fill"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "FillType", _
                "Fill Type", _
                "Fill", _
                "Fill Solid or Gradient"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "FillTypeChecked", _
                "Fill Type (CheckedButton)", _
                "Fill", _
                "Fill Solid or Gradient (CheckedButton)"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "ColorFillBlend", _
                "Blend Colors", _
                "Fill", _
                "Color and Position Arrays for the ColorBlend"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "ColorFillBlendChecked", _
                "Blend Colors (CheckedButton)", _
                "Fill", _
                "Color and Position Arrays for the ColorBlend (CheckedButton)"))

        items.Add( _
            New DesignerActionPropertyItem( _
                "ColorFillSolid", _
                "Solid Fill Color", _
                "Fill", _
                "The Color for Solid Fills"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "ColorFillSolidChecked", _
                "Solid Fill Color (CheckedButton)", _
                "Fill", _
                "The Color for Solid Fills (CheckedButton)"))

        items.Add(New DesignerActionHeaderItem("Text"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "Text", _
                "Text", _
                "Text", _
                "The Text on the Button"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "TextAlign", _
                "Text Alignment", _
                "Text", _
                "The Alignment to use on the Text"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "ForeColor", _
                "Text Color", _
                "Text", _
                "The Color of the Text"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "ForeColorChecked", _
                "Text Color (CheckedButton)", _
                "Text", _
                "The Color of the Text (CheckedButton)"))

        items.Add( _
            New DesignerActionPropertyItem( _
                "TextShadow", _
                "Shadow Color", _
                "Text", _
                "The Color of the Shadow Text"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "TextShadowChecked", _
                "Shadow Color (CheckedButton)", _
                "Text", _
                "The Color of the Shadow Text (CheckedButton)"))

        items.Add( _
            New DesignerActionPropertyItem( _
                "TextImageRelation", _
                "Text to Image Relation", _
                "Text", _
                "The Relationship of the Text to the Image"))


        items.Add(New DesignerActionHeaderItem("Border"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "BorderColor", _
                "Border Color", _
                "Border", _
                "The color of the button's Border"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "BorderShow", _
                "Show Border", _
                "Border", _
                "The show or not show the border"))

        items.Add(New DesignerActionHeaderItem("Images"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "ImageAlign", _
                "Image Alignment", _
                "Images", _
                "The Alignment for the Image"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "SideImageAlign", _
                "SideImage Alignment", _
                "Images", _
                "The Alignment for the SideImage"))
        items.Add( _
            New DesignerActionPropertyItem( _
                "SideImageBehindText", _
                "Is SideImage Behind the Text", _
                "Images", _
                ""))

        items.Add(New DesignerActionHeaderItem("Corners"))

        Dim txt As String = String.Format("UpperLeft={0}, UpperRight={1}, LowerRight={2}, LowerLeft={3}", _
        _MyButton.Corners.UpperLeft.ToString, _
        _MyButton.Corners.UpperRight.ToString, _
        _MyButton.Corners.LowerRight.ToString, _
        _MyButton.Corners.LowerLeft.ToString)
        items.Add( _
            New DesignerActionTextItem( _
                 txt, "Corners"))
        items.Add( _
            New DesignerActionMethodItem( _
                Me, _
                "AdjustCorners", _
                "Adjust Corners ", _
                "Corners", _
                "Adjust Corners", _
                True))

        Return items
    End Function

End Class

#End Region 'MyButtonActionList

