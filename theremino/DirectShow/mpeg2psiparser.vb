
Imports System.Runtime.InteropServices

Namespace BDA

#If ALLOW_UNTESTED_INTERFACES Then

	<StructLayout(LayoutKind.Sequential, Pack := 2)> _
	Public Structure ProgramElement
		Public wProgramNumber As Short
		Public wProgramMapPID As Short
	End Structure

#End If

#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("D19BDB43-405B-4a7c-A791-C89110C33165"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface ITSDT
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList, <[In]> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfTableDescriptors(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByIndex(<[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByTag(<[In]> bTag As Byte, <[In], Out> pdwCookie As DsInt, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function RegisterForNextTable(<[In]> hNextTableAvailable As IntPtr) As Integer

		<PreserveSig> _
		Function GetNextTable(<Out> ByRef ppTSDT As ITSDT) As Integer

		<PreserveSig> _
		Function RegisterForWhenCurrent(<[In]> hNextTableIsCurrent As IntPtr) As Integer

		<PreserveSig> _
		Function ConvertNextToCurrent() As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("919F24C5-7B14-42ac-A4B0-2AE08DAF00AC"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IPSITables
		<PreserveSig> _
		Function GetTable(<[In]> dwTSID As Integer, <[In]> dwTID_PID As Integer, <[In]> dwHashedVer As Integer, <[In]> dwPara4 As Integer, <Out> ByRef ppIUnknown As Object) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("BF02FB7E-9792-4e10-A68D-033A2CC246A5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IGenericDescriptor2
		Inherits IGenericDescriptor

		<PreserveSig> _
		Shadows Function Initialize(<[In]> pbDesc As IntPtr, <[In]> bCount As Integer) As Integer

		<PreserveSig> _
		Shadows Function GetTag(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Shadows Function GetLength(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Shadows Function GetBody(<Out> ByRef ppbVal As IntPtr) As Integer

        <PreserveSig()> _
        Shadows Function Initialize(ByVal pbDesc As IntPtr, ByVal wCount As Short) As Integer

        <PreserveSig()> _
        Shadows Function GetLength(ByRef pwVal As Short) As Integer

	End Interface

#End If

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("6A5918F8-A77A-4f61-AED0-5702BDCDA3E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IGenericDescriptor
        <PreserveSig()> _
        Function Initialize(<[In]()> ByVal pbDesc As IntPtr, <[In]()> ByVal bCount As Integer) As Integer

        <PreserveSig()> _
        Function GetTag(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetLength(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetBody(<Out()> ByRef ppbVal As IntPtr) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("7C6995FB-2A31-4bd7-953E-B1AD7FB7D31C"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ICAT
        <PreserveSig()> _
        Function Initialize(<[In]()> ByVal pSectionList As ISectionList, <[In]()> ByVal pMPEGData As IMpeg2Data) As Integer

        <PreserveSig()> _
        Function GetVersionNumber(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetCountOfTableDescriptors(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetTableDescriptorByIndex(<[In]()> ByVal dwIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetTableDescriptorByTag(<[In]()> ByVal bTag As Byte, <[In](), Out()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function RegisterForNextTable(<[In]()> ByVal hNextTableAvailable As IntPtr) As Integer

        <PreserveSig()> _
        Function GetNextTable(<[In]()> ByVal dwTimeout As Integer, <Out()> ByRef ppCAT As ICAT) As Integer

        <PreserveSig()> _
        Function RegisterForWhenCurrent(<[In]()> ByVal hNextTableIsCurrent As IntPtr) As Integer

        <PreserveSig()> _
        Function ConvertNextToCurrent() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("01F3B398-9527-4736-94DB-5195878E97A8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IPMT
        <PreserveSig()> _
        Function Initialize(<[In]()> ByVal pSectionList As ISectionList, <[In]()> ByVal pMPEGData As IMpeg2Data) As Integer

        <PreserveSig()> _
        Function GetProgramNumber(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetVersionNumber(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetPcrPid(<Out()> ByRef pPidVal As Short) As Integer

        <PreserveSig()> _
        Function GetCountOfTableDescriptors(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetTableDescriptorByIndex(<[In]()> ByVal dwIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetTableDescriptorByTag(<[In]()> ByVal bTag As [Byte], <[In](), Out()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetCountOfRecords(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetRecordStreamType(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetRecordElementaryPid(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pPidVal As Short) As Integer

        <PreserveSig()> _
        Function GetRecordCountOfDescriptors(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetRecordDescriptorByIndex(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal dwDescIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetRecordDescriptorByTag(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal bTag As [Byte], <[In](), Out()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function QueryServiceGatewayInfo(<Out(), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.Struct)> ByRef ppDSMCCList As DsmccElement(), <Out()> ByRef puiCount As Integer) As Integer

        <PreserveSig()> _
        Function QueryMPEInfo(<Out(), MarshalAs(UnmanagedType.LPArray, ArraySubType:=UnmanagedType.Struct)> ByRef ppMPEList As MpeElement(), <Out()> ByRef puiCount As Integer) As Integer

        <PreserveSig()> _
        Function RegisterForNextTable(<[In]()> ByVal hNextTableAvailable As IntPtr) As Integer

        <PreserveSig()> _
        Function GetNextTable(<Out()> ByRef ppPMT As IPMT) As Integer

        <PreserveSig()> _
        Function RegisterForWhenCurrent(<[In]()> ByVal hNextTableIsCurrent As IntPtr) As Integer

        <PreserveSig()> _
        Function ConvertNextToCurrent() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("6623B511-4B5F-43c3-9A01-E8FF84188060"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IPAT
        <PreserveSig()> _
        Function Initialize(<[In]()> ByVal pSectionList As ISectionList, <[In]()> ByVal pMPEGData As IMpeg2Data) As Integer

        <PreserveSig()> _
        Function GetTransportStreamId(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetVersionNumber(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetCountOfRecords(<Out()> ByRef pwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetRecordProgramNumber(<[In]()> ByVal dwIndex As Integer, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetRecordProgramMapPid(<[In]()> ByVal dwIndex As Integer, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function FindRecordProgramMapPid(<[In]()> ByVal wProgramNumber As Short, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function RegisterForNextTable(<[In]()> ByVal hNextTableAvailable As IntPtr) As Integer

        <PreserveSig()> _
        Function GetNextTable(<Out()> ByRef ppPAT As IPAT) As Integer

        <PreserveSig()> _
        Function RegisterForWhenCurrent(<[In]()> ByVal hNextTableIsCurrent As IntPtr) As Integer

        <PreserveSig()> _
        Function ConvertNextToCurrent() As Integer
    End Interface

End Namespace
