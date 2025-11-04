
Imports System.Runtime.InteropServices
Imports System.Runtime.InteropServices.ComTypes

' Note that data is returned in the memory IMMEDIATELY following this struct.
' The Size parm indicates ths size of the KSMultipleItem plus the extra bytes.
<StructLayout(LayoutKind.Sequential)> _
Public Class KSMultipleItem
    Public Size As Integer
    Public Count As Integer
End Class


<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("00000109-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IPersistStream
    Inherits IPersist

    <PreserveSig()> _
    Shadows Function GetClassID(<Out()> ByRef pClassID As Guid) As Integer

    <PreserveSig()> _
    Function IsDirty() As Integer

    <PreserveSig()> _
    Function Load(<[In]()> ByVal pStm As IStream) As Integer


    <PreserveSig()> _
    Function Save(<[In]()> ByVal pStm As IStream, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal fClearDirty As Boolean) As Integer

    <PreserveSig()> _
    Function GetSizeMax(<Out()> ByRef pcbSize As Long) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("0000010c-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IPersist
    <PreserveSig()> _
    Function GetClassID(<Out()> ByRef pClassID As Guid) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("b61178d1-a2d9-11cf-9e53-00aa00a216a1"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IKsPin
    ' The caller must free the returned structures, using the CoTaskMemFree function
    <PreserveSig()> _
    Function KsQueryMediums(ByRef ip As IntPtr) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("B196B28B-BAB4-101A-B69C-00AA00341D07"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ISpecifyPropertyPages
    <PreserveSig()> _
    Function GetPages(ByRef pPages As DsCAUUID) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("55272A00-42CB-11CE-8135-00AA004BB851"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IPropertyBag
    <PreserveSig()> _
    Function Read(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszPropName As String, <Out(), MarshalAs(UnmanagedType.Struct)> ByRef pVar As Object, <[In]()> ByVal pErrorLog As IErrorLog) As Integer

    <PreserveSig()> _
    Function Write(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszPropName As String, <[In](), MarshalAs(UnmanagedType.Struct)> ByRef pVar As Object) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("3127CA40-446E-11CE-8135-00AA004BB851"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IErrorLog

    <PreserveSig()> _
    Function AddError(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszPropName As String, <[In]()> ByVal pExcepInfo As System.Runtime.InteropServices.ComTypes.EXCEPINFO) As Integer
End Interface

