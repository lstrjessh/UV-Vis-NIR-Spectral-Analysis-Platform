
Imports System.Runtime.InteropServices

<Flags()> _
Public Enum AMPinFlowControl
    None = &H0
    Block = &H1
End Enum

<Flags()> _
Public Enum AMGraphConfigReconnect
    None = &H0
    DirectConnect = &H1
    CacheRemovedFilters = &H2
    UseOnlyCachedFilters = &H4
End Enum

<Flags()> _
Public Enum AMFilterFlags
    None = &H0
    Removable = &H1
End Enum

<Flags()> _
Public Enum RemFilterFlags
    None = &H0
    LeaveConnected = &H1
End Enum



<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("c56e9858-dbf3-4f6b-8119-384af2060deb"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IPinFlowControl
		' HEVENT
	<PreserveSig> _
	Function Block(<[In]> dwBlockFlags As AMPinFlowControl, <[In]> hEvent As IntPtr) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("DCFBDCF6-0DC2-45f5-9AB2-7C330EA09C29"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IFilterChain
	<PreserveSig> _
	Function StartChain(<[In]> pStartFilter As IBaseFilter, <[In]> pEndFilter As IBaseFilter) As Integer

	<PreserveSig> _
	Function PauseChain(<[In]> pStartFilter As IBaseFilter, <[In]> pEndFilter As IBaseFilter) As Integer

	<PreserveSig> _
	Function StopChain(<[In]> pStartFilter As IBaseFilter, <[In]> pEndFilter As IBaseFilter) As Integer

	<PreserveSig> _
	Function RemoveChain(<[In]> pStartFilter As IBaseFilter, <[In]> pEndFilter As IBaseFilter) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("ade0fd60-d19d-11d2-abf6-00a0c905f375"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IGraphConfigCallback
	' PVOID
	<PreserveSig> _
	Function Reconfigure(pvContext As IntPtr, dwFlags As Integer) As Integer

End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("03A1EB8E-32BF-4245-8502-114D08A9CB88"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IGraphConfig
	' can be NULL
	' HANDLE
	<PreserveSig> _
	Function Reconnect(<[In]> pOutputPin As IPin, <[In]> pInputPin As IPin, <[In], MarshalAs(UnmanagedType.LPStruct)> pmtFirstConnection As AMMediaType, <[In]> pUsingFilter As IBaseFilter, <[In]> hAbortEvent As IntPtr, <[In]> dwFlags As AMGraphConfigReconnect) As Integer

	' PVOID
    ' HANDLE
	<PreserveSig> _
	Function Reconfigure(<[In]> pCallback As IGraphConfigCallback, <[In]> pvContext As IntPtr, <[In]> dwFlags As Integer, <[In]> hAbortEvent As IntPtr) As Integer

	<PreserveSig> _
	Function AddFilterToCache(<[In]> pFilter As IBaseFilter) As Integer

	<PreserveSig> _
	Function EnumCacheFilter(<Out> ByRef pEnum As IEnumFilters) As Integer

	<PreserveSig> _
	Function RemoveFilterFromCache(<[In]> pFilter As IBaseFilter) As Integer

	<PreserveSig> _
	Function GetStartTime(<Out> ByRef prtStart As Long) As Integer

    ' HANDLE
	<PreserveSig> _
	Function PushThroughData(<[In]> pOutputPin As IPin, <[In]> pConnection As IPinConnection, <[In]> hEventAbort As IntPtr) As Integer

	<PreserveSig> _
	Function SetFilterFlags(<[In]> pFilter As IBaseFilter, <[In]> dwFlags As AMFilterFlags) As Integer

	<PreserveSig> _
	Function GetFilterFlags(<[In]> pFilter As IBaseFilter, <Out> ByRef pdwFlags As AMFilterFlags) As Integer

	<PreserveSig> _
	Function RemoveFilterEx(<[In]> pFilter As IBaseFilter, Flags As RemFilterFlags) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("4a9a62d3-27d4-403d-91e9-89f540e55534"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IPinConnection
	<PreserveSig> _
	Function DynamicQueryAccept(<[In], MarshalAs(UnmanagedType.LPStruct)> pmt As AMMediaType) As Integer

    ' HEVENT
	<PreserveSig> _
	Function NotifyEndOfStream(<[In]> hNotifyEvent As IntPtr) As Integer

	<PreserveSig> _
	Function IsEndPin() As Integer

	<PreserveSig> _
	Function DynamicDisconnect() As Integer
End Interface

