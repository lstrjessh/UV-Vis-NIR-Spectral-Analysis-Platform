

' ======================================================================================================
'   MyListView   -   implements  "Header colors" and "In-place subitem editing"
' ======================================================================================================

Imports System.ComponentModel
Imports System.Runtime.InteropServices
Imports System.Drawing

Public Class MyListView
    Inherits System.Windows.Forms.ListView

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

    <System.Diagnostics.DebuggerStepThrough()> _
    Private Sub InitializeComponent()
        components = New System.ComponentModel.Container
    End Sub
    ' =================================================================================== 
    '      required by Designer   -    do not modify with the code editor
    ' ===================================================================================




    ' =================================================================================================
    '   LIST-VIEW DOUBLE BUFFERING  
    '   ( working only if project/propertyes/xpstyles are enabled )
    ' =================================================================================================
    Public Enum LVS_EX
        LVS_EX_GRIDLINES = &H1
        LVS_EX_SUBITEMIMAGES = &H2
        LVS_EX_CHECKBOXES = &H4
        LVS_EX_TRACKSELECT = &H8
        LVS_EX_HEADERDRAGDROP = &H10
        LVS_EX_FULLROWSELECT = &H20
        LVS_EX_ONECLICKACTIVATE = &H40
        LVS_EX_TWOCLICKACTIVATE = &H80
        LVS_EX_FLATSB = &H100
        LVS_EX_REGIONAL = &H200
        LVS_EX_INFOTIP = &H400
        LVS_EX_UNDERLINEHOT = &H800
        LVS_EX_UNDERLINECOLD = &H1000
        LVS_EX_MULTIWORKAREAS = &H2000
        LVS_EX_LABELTIP = &H4000
        LVS_EX_BORDERSELECT = &H8000
        LVS_EX_DOUBLEBUFFER = &H10000
        LVS_EX_HIDELABELS = &H20000
        LVS_EX_SINGLEROW = &H40000
        LVS_EX_SNAPTOGRID = &H80000
        LVS_EX_SIMPLESELECT = &H100000
    End Enum
    Private Enum LVM
        LVM_FIRST = &H1000
        LVM_SETEXTENDEDLISTVIEWSTYLE = (LVM_FIRST + 54)
        LVM_GETEXTENDEDLISTVIEWSTYLE = (LVM_FIRST + 55)
    End Enum
    Private Declare Auto Function SendMessage Lib "user32.dll" (ByVal handle As IntPtr, _
                                                                ByVal messg As Int32, _
                                                                ByVal wparam As Int32, _
                                                                ByVal lparam As Int32) As Int32
    Public styles As LVS_EX
    Public Sub SetExtStyle_DoubleBuffering()
        If Not OperatingSystemIsWindows Then Return
        styles = CType(SendMessage(Me.Handle, CInt(LVM.LVM_GETEXTENDEDLISTVIEWSTYLE), 0, 0), LVS_EX)
        styles = styles Or LVS_EX.LVS_EX_DOUBLEBUFFER
        SendMessage(Me.Handle, CInt(LVM.LVM_SETEXTENDEDLISTVIEWSTYLE), 0, CInt(styles))
    End Sub
    Public Sub SetExtStyle_BorderSelect()
        If Not OperatingSystemIsWindows Then Return
        styles = CType(SendMessage(Me.Handle, CInt(LVM.LVM_GETEXTENDEDLISTVIEWSTYLE), 0, 0), LVS_EX)
        styles = styles Or LVS_EX.LVS_EX_BORDERSELECT
        SendMessage(Me.Handle, CInt(LVM.LVM_SETEXTENDEDLISTVIEWSTYLE), 0, CInt(styles))
    End Sub
    Public Sub SetExtStyle(ByVal exStyle As LVS_EX)
        If Not OperatingSystemIsWindows Then Return
        styles = CType(SendMessage(Me.Handle, CInt(LVM.LVM_GETEXTENDEDLISTVIEWSTYLE), 0, 0), LVS_EX)
        styles = styles Or exStyle
        SendMessage(Me.Handle, CInt(LVM.LVM_SETEXTENDEDLISTVIEWSTYLE), 0, CInt(styles))
    End Sub



    ' =================================================================================== 
    '    INITIALIZATIONS
    ' ===================================================================================

    Private _Header_BackColor1 As Color
    Private _Header_BackColor2 As Color
    Private _Header_ForeColor As Color
    Private _Header_ShadowColor As Color
    Private _Header_Font As Font


    Private Sub Initialize()
        OwnerDraw = True
        _Header_BackColor1 = Color.FromArgb(255, 255, 200)
        _Header_BackColor2 = Color.FromArgb(200, 200, 200)
        _Header_ForeColor = Color.FromArgb(100, 80, 0)
        _Header_ShadowColor = Color.FromArgb(120, 120, 120)
        _Header_Font = New Font("Tahoma", 10, FontStyle.Regular)
    End Sub

    <Category("Headers"), _
    Description("Get or Set the Header BackColor 1")> _
    Public Property Header_BackColor1() As Color
        Get
            Return _Header_BackColor1
        End Get
        Set(ByVal value As Color)
            If value <> _Header_BackColor1 Then
                _Header_BackColor1 = value
                Refresh()
            End If
        End Set
    End Property

    <Category("Headers"), _
    Description("Get or Set the Header BackColor 2")> _
    Public Property Header_BackColor2() As Color
        Get
            Return _Header_BackColor2
        End Get
        Set(ByVal value As Color)
            If value <> _Header_BackColor2 Then
                _Header_BackColor2 = value
                Refresh()
            End If
        End Set
    End Property

    <Category("Headers"), _
    Description("Get or Set the Header ForeColor")> _
    Public Property Header_ForeColor() As Color
        Get
            Return _Header_ForeColor
        End Get
        Set(ByVal value As Color)
            If value <> _Header_ForeColor Then
                _Header_ForeColor = value
                Refresh()
            End If
        End Set
    End Property

    <Category("Headers"), _
    Description("Get or Set the Header ShadowColor")> _
    Public Property Header_ShadowColor() As Color
        Get
            Return _Header_ShadowColor
        End Get
        Set(ByVal value As Color)
            If value <> _Header_ShadowColor Then
                _Header_ShadowColor = value
                Refresh()
            End If
        End Set
    End Property

    <Category("Headers"), _
    Description("Get or Set the Header Font")> _
    Public Property Header_Font() As Font
        Get
            Return _Header_Font
        End Get
        Set(ByVal value As Font)
            _Header_Font = value
            Refresh()
        End Set
    End Property


    Private _ShadowColor As Color = Color.Transparent
    <Category("Appearance"), _
    Description("Get or Set the ShadowColor")> _
    Public Property ShadowColor() As Color
        Get
            Return _ShadowColor
        End Get
        Set(ByVal Value As Color)
            If Value <> _ShadowColor Then
                _ShadowColor = Value
                Refresh()
            End If
        End Set
    End Property





    ' =================================================================================== 
    '    COLUMN HEADERS
    ' ===================================================================================

    Private Sub MyDrawColumnHeader(ByVal sender As Object, ByVal e As System.Windows.Forms.DrawListViewColumnHeaderEventArgs) Handles Me.DrawColumnHeader
        '
        Dim g As Graphics = e.Graphics
        Dim x1 As Integer = e.Bounds.Left
        Dim x2 As Integer = e.Bounds.Right
        Dim y1 As Integer = e.Bounds.Top
        Dim y2 As Integer = e.Bounds.Bottom
        '
        If e.Bounds.Width > 0 Then
            g.FillRectangle(New System.Drawing.Drawing2D.LinearGradientBrush(e.Bounds, _
                                                                             _Header_BackColor1, _
                                                                             _Header_BackColor2, _
                                                                             Drawing2D.LinearGradientMode.Vertical), _
                                                                             e.Bounds)
            '
            g.DrawLine(Pens.Gold, x2 - 1, y1, x2 - 1, y2)
            g.DrawLine(Pens.Gold, x2, y1, x2, y2)
            g.DrawLine(Pens.DarkSlateBlue, x1, y2 - 1, x2, y2 - 1)
            '
            Dim sf As StringFormat = New StringFormat
            sf.Alignment = StringAlignment.Center

            Dim dy As Int32 = (e.Bounds.Height - _Header_Font.Height) \ 2 - 1

            Dim textrec As RectangleF = New RectangleF(e.Bounds.X - 2, e.Bounds.Y + dy, e.Bounds.Width + 2, e.Bounds.Height)
            ' ----------------------------------------------------------------- draw the Shadow
            If _Header_ShadowColor.A <> 0 Then
                textrec.Offset(1, 1)
                g.DrawString(e.Header.Text, _Header_Font, _
                                      New SolidBrush(_Header_ShadowColor), textrec, sf)
                textrec.Offset(-1, -1)
            End If

            ' ----------------------------------------------------------------- draw the Text
            g.DrawString(e.Header.Text, _
                         _Header_Font, _
                         New SolidBrush(_Header_ForeColor), _
                         textrec, sf)
        End If
    End Sub

    Private Sub MyDrawItem(ByVal sender As Object, ByVal e As System.Windows.Forms.DrawListViewItemEventArgs) Handles Me.DrawItem
        e.DrawDefault = True
    End Sub
    Private Sub MyDrawSubItem(ByVal sender As Object, ByVal e As System.Windows.Forms.DrawListViewSubItemEventArgs) Handles Me.DrawSubItem
        e.DrawDefault = True
    End Sub

End Class









'' #####################################################################################################
''   EDIT SUBITEMS -  INIT and EVENTS -  USAGE EXAMPLES
'' #####################################################################################################

'Private Sub InitSubitemEditing()
'    ' --------------------------------------------------------------------- A control for each column
'    '                                                                       ( nothing for non editable columns )
'    ListView1.SetEditors(New Control() {TextBox_EditText, _
'                             TextBox_EditText, _
'                             Nothing, _
'                             NumericUpDown1, _
'                             DateTimePicker1})

'    ' --------------------------------------------------------------------- useful events
'    AddHandler ListView1.SubItemClicked, AddressOf Me.ListView1_SubItemClicked
'    AddHandler ListView1.SubItemEndEditing, AddressOf Me.ListView1_SubItemEndEditing

'    ' --------------------------------------------------------------------- also edit with the double click
'    ListView1.DoubleClickActivation = True
'End Sub

'Private Sub ListView1_SubItemClicked(ByVal sender As Object, ByVal e As SubItemEventArgs)
'    ' ------------------------------------------------------------------------ pre edit
'    'If (e.SubItem = 3) Then
'    'End If
'    ' ------------------------------------------------------------------------ start editing
'    ListView1.StartEditing(e.Item, e.SubItem)
'End Sub

'Private Sub ListView1_SubItemEndEditing(ByVal sender As Object, ByVal e As SubItemEndEditingEventArgs)
'    ' ------------------------------------------------------------------------ post edit
'    'If (e.SubItem = 3) Then
'    'End If
'End Sub
'' #####################################################################################################
''   EDIT SUBITEMS -  EXAMPLES - END
'' #####################################################################################################







' #####################################################################################################
'   REM FROM HERE TO THE END IF NOT USING EDIT
' #####################################################################################################


'' =================================================================================== 
''    SUB-ITEMS  IN-PLACE  EDITING
'' ===================================================================================

'Partial Public Class MyListView
'    ' ---------------------------------------------------------------- MyListView messages
'    Private Const LVM_FIRST As Int32 = 4096
'    Private Const LVM_GETCOLUMNORDERARRAY As Int32 = (LVM_FIRST + 59)

'    ' ---------------------------------------------------------------- Windows Messages that will abort editing
'    Private Const WM_HSCROLL As Int32 = 276
'    Private Const WM_VSCROLL As Int32 = 277
'    Private Const WM_SIZE As Int32 = 5
'    Private Const WM_NOTIFY As Int32 = 78
'    Private Const HDN_FIRST As Int32 = -300
'    Private Const HDN_BEGINDRAG As Int32 = (HDN_FIRST - 10)
'    Private Const HDN_ITEMCHANGINGA As Int32 = (HDN_FIRST - 0)
'    Private Const HDN_ITEMCHANGINGW As Int32 = (HDN_FIRST - 20)

'    ' ---------------------------------------------------------------- Double click activation flag
'    Private _doubleClickActivation As Boolean = False

'    ' ---------------------------------------------------------------- The control performing the actual editing
'    Private _editingControl As Control

'    ' ---------------------------------------------------------------- The LVI being edited
'    Private _editItem As ListViewItem

'    ' ---------------------------------------------------------------- The SubItem being edited
'    Private _editSubItem As Int32

'    ' ---------------------------------------------------------------- The "EditingControls" array
'    Private _editingControls As Control()


'    ' ---------------------------------------------------------------- Is a double click required to start editing a cell?
'    Public Property DoubleClickActivation() As Boolean
'        Get
'            Return _doubleClickActivation
'        End Get
'        Set(ByVal value As Boolean)
'            _doubleClickActivation = value
'        End Set
'    End Property

'    Public Event SubItemClicked As SubItemEventHandler

'    Public Event SubItemBeginEditing As SubItemEventHandler

'    Public Event SubItemEndEditing As SubItemEndEditingEventHandler

'    Private Overloads Declare Auto Function SendMessage Lib "user32.dll" (ByVal hWnd As IntPtr, ByVal msg As Int32, ByVal wPar As IntPtr, ByVal lPar As IntPtr) As IntPtr

'    Private Overloads Declare Auto Function SendMessage Lib "user32.dll" (ByVal hWnd As IntPtr, ByVal msg As Int32, ByVal len As Int32, ByRef order() As Int32) As IntPtr



'    ' --------------------------------------------------- Prepare the "Editors" ( one for each column )
'    Public Sub SetEditors(ByVal EditorsArray() As Control)
'        _editingControls = EditorsArray
'    End Sub



'    ' --------------------------------------------------- Retrieve the order in which columns appear
'    '                                                      returns >> current display order of column indices
'    Public Function GetColumnOrder() As Int32()
'        Dim lPar As IntPtr = Marshal.AllocHGlobal((Marshal.SizeOf(GetType(System.Int32)) * Columns.Count))
'        Dim res As IntPtr = SendMessage(Handle, LVM_GETCOLUMNORDERARRAY, New IntPtr(Columns.Count), lPar)
'        If (res.ToInt32 = 0) Then
'            Marshal.FreeHGlobal(lPar)
'            Return Nothing
'        End If
'        Dim order() As Int32 = New Int32((Columns.Count) - 1) {}
'        Marshal.Copy(lPar, order, 0, Columns.Count)
'        Marshal.FreeHGlobal(lPar)
'        Return order
'    End Function


'    ' ---------------------------------------------------- Find ListViewItem and SubItem Index at position (x,y)
'    '
'    '                                                           "x" relative to MyListView
'    '                                                           "y" relative to MyListView
'    '                                                           "item" Item at position (x,y)
'    '                                                           returns: SubItem index
'    '
'    Public Function GetSubItemAt(ByVal x As Int32, ByVal y As Int32, ByRef item As ListViewItem) As Int32
'        item = Me.GetItemAt(x, y)
'        If (Not (item) Is Nothing) Then
'            Dim order() As Int32 = GetColumnOrder()
'            Dim lviBounds As Rectangle
'            Dim subItemX As Int32
'            lviBounds = item.GetBounds(ItemBoundsPortion.Entire)
'            subItemX = lviBounds.Left
'            For i As Int32 = 0 To order.Length - 1
'                Dim h As ColumnHeader = Me.Columns(order(i))
'                If (x < subItemX + h.Width) Then
'                    Return h.Index
'                End If
'                subItemX = (subItemX + h.Width)
'            Next
'        End If
'        Return -1
'    End Function


'    ' ----------------------------------------------------------- Get bounds for a SubItem
'    '
'    '                                                              "Item" Target ListViewItem 
'    '                                                              "SubItem" Target SubItem index</param>
'    '                                                              returns: Bounds of SubItem (relative to MyListView)
'    '
'    Public Function GetSubItemBounds(ByVal Item As ListViewItem, ByVal SubItem As Int32) As Rectangle
'        Dim order() As Int32 = GetColumnOrder()
'        Dim subItemRect As Rectangle = Rectangle.Empty
'        If (SubItem >= order.Length) Then
'            Throw New IndexOutOfRangeException(("SubItem " & SubItem & " out of range"))
'        End If
'        If (Item Is Nothing) Then
'            Throw New ArgumentNullException("Item")
'        End If
'        Dim lviBounds As Rectangle = Item.GetBounds(ItemBoundsPortion.Entire)
'        Dim subItemX As Int32 = lviBounds.Left
'        Dim col As ColumnHeader
'        Dim i As Int32
'        For i = 0 To order.Length - 1
'            col = Me.Columns(order(i))
'            If col.Index = SubItem Then
'                Exit For
'            End If
'            subItemX = subItemX + col.Width
'        Next
'        subItemRect = New Rectangle(subItemX, lviBounds.Top, Me.Columns(order(i)).Width, lviBounds.Height)
'        Return subItemRect
'    End Function

'    Protected Overrides Sub WndProc(ByRef msg As Message)
'        Select Case (msg.Msg)
'            Case WM_VSCROLL, WM_HSCROLL, WM_SIZE
'                EndEditing(False)
'            Case WM_NOTIFY
'                ' Look for WM_NOTIFY of events that might also change the
'                ' editor's position/size: Column reordering or resizing
'                Dim h As NMHDR = CType(Marshal.PtrToStructure(msg.LParam, GetType(NMHDR)), NMHDR)
'                If (h.code = HDN_BEGINDRAG) _
'                            OrElse ((h.code = HDN_ITEMCHANGINGA) _
'                            OrElse (h.code = HDN_ITEMCHANGINGW)) Then
'                    EndEditing(False)
'                End If
'        End Select
'        MyBase.WndProc(msg)
'    End Sub

'    Protected Overrides Sub OnMouseUp(ByVal e As System.Windows.Forms.MouseEventArgs)
'        MyBase.OnMouseUp(e)
'        ' --------------------------------------------- right mouse click activation is always active
'        If e.Button = Windows.Forms.MouseButtons.Right Then
'            EditSubitemAt(New Point(e.X, e.Y))
'        End If
'    End Sub

'    Protected Overrides Sub OnDoubleClick(ByVal e As EventArgs)
'        MyBase.OnDoubleClick(e)
'        If Not DoubleClickActivation Then Return
'        Dim pt As Point = Me.PointToClient(Cursor.Position)
'        EditSubitemAt(pt)
'    End Sub


'    ' ------------------------------------------------------------- Fire SubItemClicked
'    '                                                                "p" Point of click/doubleclick
'    Private Sub EditSubitemAt(ByVal p As Point)
'        Dim item As ListViewItem = Nothing
'        Dim idx As Int32 = GetSubItemAt(p.X, p.Y, item)
'        If idx >= 0 Then
'            OnSubItemClicked(New SubItemEventArgs(item, idx))
'        End If
'    End Sub

'    Protected Sub OnSubItemBeginEditing(ByVal e As SubItemEventArgs)
'        RaiseEvent SubItemBeginEditing(Me, e)
'    End Sub

'    Protected Sub OnSubItemEndEditing(ByVal e As SubItemEndEditingEventArgs)
'        RaiseEvent SubItemEndEditing(Me, e)
'    End Sub

'    Protected Sub OnSubItemClicked(ByVal e As SubItemEventArgs)
'        RaiseEvent SubItemClicked(Me, e)
'    End Sub


'    ' ---------------------------------------------------------------- Begin in-place editing of given cell
'    '
'    '                                                                   "c" Control used as cell editor
'    '                                                                   "Item" ListViewItem to edit
'    '                                                                   "SubItem" SubItem index to edit
'    '
'    Public Sub StartEditing(ByVal Item As ListViewItem, ByVal SubItem As Int32)

'        If _editingControls Is Nothing Then Exit Sub
'        If SubItem < _editingControls.GetLowerBound(0) Or SubItem > _editingControls.GetUpperBound(0) Then Exit Sub
'        '
'        Dim c As Control = _editingControls(SubItem)
'        If c Is Nothing Then Exit Sub

'        OnSubItemBeginEditing(New SubItemEventArgs(Item, SubItem))
'        Dim rcSubItem As Rectangle = GetSubItemBounds(Item, SubItem)

'        ' --------------------------------------------------- if Left edge of SubItem not visible - adjust rectangle position and width
'        If rcSubItem.X < 0 Then
'            rcSubItem.Width = rcSubItem.Width + rcSubItem.X
'            rcSubItem.X = 0
'        End If

'        ' --------------------------------------------------- if Right edge of SubItem not visible - adjust rectangle width
'        If rcSubItem.X + rcSubItem.Width > Me.Width Then
'            rcSubItem.Width = Me.Width - rcSubItem.Left
'        End If
'        ' --------------------------------------------------- Subitem bounds are relative to the location of the MyListView!
'        rcSubItem.Offset(Left() + 5, Top - 1)

'        ' --------------------------------------------------- In case the editing control and the listview are on different parents,
'        '                                                      account for different origins
'        Dim origin As Point = New Point(0, 0)
'        Dim lvOrigin As Point = Me.Parent.PointToScreen(origin)
'        Dim ctlOrigin As Point = c.Parent.PointToScreen(origin)
'        rcSubItem.Offset((lvOrigin.X - ctlOrigin.X), (lvOrigin.Y - ctlOrigin.Y))

'        ' --------------------------------------------------- Position and show editor

'        c.Bounds = rcSubItem
'        c.Font = Item.SubItems(SubItem).Font
'        c.Text = Item.SubItems(SubItem).Text
'        c.Visible = True
'        c.BringToFront()
'        c.Focus()
'        _editingControl = c
'        AddHandler _editingControl.Leave, AddressOf Me._editControl_Leave
'        AddHandler _editingControl.KeyPress, AddressOf Me._editControl_KeyPress
'        _editItem = Item
'        _editSubItem = SubItem
'    End Sub

'    Private Sub _editControl_Leave(ByVal sender As Object, ByVal e As EventArgs)
'        ' ---------------------------------------------------- cell editor losing focus
'        EndEditing(True)
'    End Sub

'    Private Sub _editControl_KeyPress(ByVal sender As Object, ByVal e As System.Windows.Forms.KeyPressEventArgs)
'        Select Case (e.KeyChar)
'            Case ChrW(Keys.Escape)
'                EndEditing(False)
'                Exit Select
'            Case ChrW(Keys.Enter)
'                EndEditing(True)
'                Exit Select
'        End Select
'    End Sub


'    ' ------------------------------------------------------ Accept or discard current value of cell editor control
'    '
'    '     "AcceptChanges" Use the _editingControl's Text as new SubItem text or discard changes?
'    '
'    ' --------------------------------------------------------------------------------------------------------------
'    Public Sub EndEditing(ByVal AcceptChanges As Boolean)
'        If (_editingControl Is Nothing) Then
'            Return
'        End If
'        Dim e As SubItemEndEditingEventArgs = New SubItemEndEditingEventArgs(_editItem, _
'                                                                             _editSubItem, _
'                                                                             If(AcceptChanges, _
'                                                                                _editingControl.Text, _
'                                                                                _editItem.SubItems(_editSubItem).Text), _
'                                                                             Not AcceptChanges)
'        OnSubItemEndEditing(e)
'        _editItem.SubItems(_editSubItem).Text = e.DisplayText
'        RemoveHandler _editingControl.Leave, AddressOf Me._editControl_Leave
'        RemoveHandler _editingControl.KeyPress, AddressOf Me._editControl_KeyPress
'        _editingControl.Visible = False
'        _editingControl = Nothing
'        _editItem = Nothing
'        _editSubItem = -1
'    End Sub


'    ' ---------------------------------------------------------------- MessageHeader for WM_NOTIFY
'    Private Structure NMHDR
'        Public hwndFrom As IntPtr
'        Public idFrom As Int32
'        Public code As Int32
'    End Structure

'End Class





'' ----------------------------------------------------------- Event Handler for SubItem events
'Public Delegate Sub SubItemEventHandler(ByVal sender As Object, ByVal e As SubItemEventArgs)

'' ----------------------------------------------------------- Event Handler for SubItemEndEditing events
'Public Delegate Sub SubItemEndEditingEventHandler(ByVal sender As Object, ByVal e As SubItemEndEditingEventArgs)


'' ----------------------------------------------------------- Event Args for SubItemClicked event
'Public Class SubItemEventArgs
'    Inherits EventArgs

'    Private _subItemIndex As Int32 = -1

'    Private _item As ListViewItem = Nothing

'    Public Sub New(ByVal item As ListViewItem, ByVal subItem As Int32)
'        MyBase.New()
'        _subItemIndex = subItem
'        _item = item
'    End Sub

'    Public ReadOnly Property SubItem() As Int32
'        Get
'            Return _subItemIndex
'        End Get
'    End Property

'    Public ReadOnly Property Item() As ListViewItem
'        Get
'            Return _item
'        End Get
'    End Property
'End Class


'' --------------------------------------------------------- Event Args for SubItemEndEditingClicked event
'Public Class SubItemEndEditingEventArgs
'    Inherits SubItemEventArgs

'    Private _text As String = String.Empty

'    Private _cancel As Boolean = True

'    Public Sub New(ByVal item As ListViewItem, ByVal subItem As Int32, ByVal display As String, ByVal cancel As Boolean)
'        MyBase.New(item, subItem)
'        _text = display
'        _cancel = cancel
'    End Sub

'    Public Property DisplayText() As String
'        Get
'            Return _text
'        End Get
'        Set(ByVal value As String)
'            _text = value
'        End Set
'    End Property

'    Public Property Cancel() As Boolean
'        Get
'            Return _cancel
'        End Get
'        Set(ByVal value As Boolean)
'            _cancel = value
'        End Set
'    End Property
'End Class

