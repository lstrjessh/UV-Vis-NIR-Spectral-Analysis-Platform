
Imports System.Runtime.InteropServices

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("52d6f586-9f0f-4824-8fc8-e32ca04930c2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDMOWrapperFilter
	<PreserveSig> _
	Function Init(<[In], MarshalAs(UnmanagedType.LPStruct)> clsidDMO As Guid, <[In], MarshalAs(UnmanagedType.LPStruct)> catDMO As Guid) As Integer
End Interface

