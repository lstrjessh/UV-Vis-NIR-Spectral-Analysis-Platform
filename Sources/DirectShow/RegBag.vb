
Imports System.Runtime.InteropServices

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("8A674B48-1F63-11d3-B64C-00C04F79498E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ICreatePropBagOnRegKey
	<PreserveSig> _
	Function Create(<[In]> hkey As IntPtr, <[In], MarshalAs(UnmanagedType.LPWStr)> subkey As String, <[In]> ulOptions As Integer, <[In]> samDesired As Integer, <[In], MarshalAs(UnmanagedType.LPStruct)> iid As Guid, <Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppBag As Object) As Integer
End Interface

