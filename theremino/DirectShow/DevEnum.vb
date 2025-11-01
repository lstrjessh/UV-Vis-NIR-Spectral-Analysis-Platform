
Imports System.Runtime.InteropServices

<Flags> _
Public Enum CDef
	None = 0
	ClassDefault = &H1
	BypassClassManager = &H2
	ClassLegacy = &H4
	MeritAboveDoNotUse = &H8
	DevmonCMGRDevice = &H10
	DevmonDMO = &H20
	DevmonPNPDevice = &H40
	DevmonFilter = &H80
	DevmonSelectiveMask = &Hf0
End Enum

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ICreateDevEnum
    <PreserveSig()> _
    Function CreateClassEnumerator(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pType As Guid, <Out()> ByRef ppEnumMoniker As ComTypes.IEnumMoniker, <[In]()> ByVal dwFlags As CDef) As Integer
End Interface

