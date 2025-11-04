
Imports System.Runtime.InteropServices

#If ALLOW_UNTESTED_INTERFACES Then

    Public Enum AMPlayListItemFlags
	    CanSkip = &H1
	    CanBind = &H2
    End Enum

    <Flags> _
    Public Enum AMPlayListFlags
	    None = 0
	    StartInScanMode = &H1
	    ForceBanner = &H2
    End Enum

#End If


#If ALLOW_UNTESTED_INTERFACES Then

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868ff-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMPlayListItem
	Function GetFlags(ByRef pdwFlags As AMPlayListItemFlags) As Integer

	Function GetSourceCount(ByRef pdwSources As Integer) As Integer

	Function GetSourceURL(dwSourceIndex As Integer, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrURL As String) As Integer

	Function GetSourceStart(dwSourceIndex As Integer, ByRef prtStart As Long) As Integer

	Function GetSourceDuration(dwSourceIndex As Integer, ByRef prtDuration As Long) As Integer

	Function GetSourceStartMarker(dwSourceIndex As Integer, ByRef pdwMarker As Integer) As Integer

	Function GetSourceEndMarker(dwSourceIndex As Integer, ByRef pdwMarker As Integer) As Integer

	Function GetSourceStartMarkerName(dwSourceIndex As Integer, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrStartMarker As String) As Integer

	Function GetSourceEndMarkerName(dwSourceIndex As Integer, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrEndMarker As String) As Integer

	Function GetLinkURL(<MarshalAs(UnmanagedType.BStr)> ByRef pbstrURL As String) As Integer

	Function GetScanDuration(dwSourceIndex As Integer, ByRef prtScanDuration As Long) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("56a868fe-0ad4-11ce-b03a-0020af0ba770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMPlayList
	Function GetFlags(ByRef pdwFlags As AMPlayListFlags) As Integer

	Function GetItemCount(ByRef pdwItems As Integer) As Integer

	Function GetItem(dwItemIndex As Integer, ByRef ppItem As IAMPlayListItem) As Integer

	Function GetNamedEvent(pwszEventName As String, dwItemIndex As Integer, ByRef ppItem As IAMPlayListItem, ByRef pdwFlags As AMPlayListItemFlags) As Integer

	Function GetRepeatInfo(ByRef pdwRepeatCount As Integer, ByRef pdwRepeatStart As Integer, ByRef pdwRepeatEnd As Integer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("4C437B91-6E9E-11d1-A704-006097C4E476"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ISpecifyParticularPages
	Function GetPages(<[In], MarshalAs(UnmanagedType.LPStruct)> guidWhatPages As Guid, ByRef pPages As DsCAUUID) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("02EF04DD-7580-11d1-BECE-00C04FB6E937"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IAMRebuild
	Function RebuildNow() As Integer
End Interface

#End If

