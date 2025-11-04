
Imports System.Runtime.InteropServices

Public Enum MPEGRequestType
    ' Fields
    PES_STREAM = 6
    SECTION = 1
    SECTION_ASYNC = 2
    SECTIONS_STREAM = 5
    TABLE = 3
    TABLE_ASYNC = 4
    TS_STREAM = 7
    START_MPE_STREAM = 8
    UNKNOWN = 0
End Enum

Public Enum MPEGContextType
    BCSDeMux = 0
    WinSock = 1
End Enum

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Structure MPEGPacketList
    Public wPacketCount As Short
    Public PacketList As IntPtr
End Structure

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Structure DSMCCFilterOptions
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyProtocol As Boolean
    Public Protocol As Byte
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyType As Boolean
    Public Type As Byte
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyMessageId As Boolean
    Public MessageId As Short
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyTransactionId As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public fUseTrxIdMessageIdMask As Boolean
    Public TransactionId As Integer
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyModuleVersion As Boolean
    Public ModuleVersion As Byte
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyBlockNumber As Boolean
    Public BlockNumber As Short
    <MarshalAs(UnmanagedType.Bool)> _
    Public fGetModuleCall As Boolean
    Public NumberOfBlocksInModule As Short
End Structure

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Structure ATSCFilterOptions
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyEtmId As Boolean
    Public EtmId As Integer
End Structure

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Class MPEG2Filter
    Public bVersionNumber As Byte
    Public wFilterSize As Short
    <MarshalAs(UnmanagedType.Bool)> _
    Public fUseRawFilteringBits As Boolean
    <MarshalAs(UnmanagedType.ByValArray, SizeConst:=16)> _
    Public Filter As Byte()
    <MarshalAs(UnmanagedType.ByValArray, SizeConst:=16)> _
    Public Mask As Byte()
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyTableIdExtension As Boolean
    Public TableIdExtension As Short
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyVersion As Boolean
    Public Version As Byte
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifySectionNumber As Boolean
    Public SectionNumber As Byte
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyCurrentNext As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public fNext As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyDsmccOptions As Boolean
    <MarshalAs(UnmanagedType.Struct)> _
    Public Dsmcc As DSMCCFilterOptions
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSpecifyAtscOptions As Boolean
    <MarshalAs(UnmanagedType.Struct)> _
    Public Atsc As ATSCFilterOptions
End Class

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Class DVB_EIT_FILTER_OPTIONS
    <MarshalAs(UnmanagedType.Bool)> _
    Private fSpecifySegment As Boolean
    Private bSegment As Byte
End Class

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Class MPEG2Filter2
    Inherits MPEG2Filter
    <MarshalAs(UnmanagedType.Bool)> _
    Private fSpecifyDvbEitOptions As Boolean
    Private DvbEit As DVB_EIT_FILTER_OPTIONS
End Class

<StructLayout(LayoutKind.Explicit, Pack:=1)> _
Public Structure MPEGContextUnion
    ' Fields
    <FieldOffset(0)> _
    Public Demux As BCSDeMux
    <FieldOffset(0)> _
    Public Winsock As MPEGWinSock
End Structure

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Structure BCSDeMux
    Public AVMGraphId As Integer
End Structure

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Structure MPEGWinSock
    Public AVMGraphId As Integer
End Structure

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Class MPEGContext
    Public Type As MPEGContextType
    Public U As MPEGContextUnion
End Class

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Class MPEGStreamBuffer
    '[MarshalAs(UnmanagedType.Error)]
    Public hr As Integer
    Public dwDataBufferSize As Integer
    Public dwSizeOfDataRead As Integer
    Public pDataBuffer As IntPtr
End Class



#If ALLOW_UNTESTED_INTERFACES Then

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("BDCDD913-9ECD-4fb2-81AE-ADF747EA75A5")> _
Public Interface IMpeg2TableFilter
	<PreserveSig> _
	Function AddPID(pid As Short) As Integer

	<PreserveSig> _
	Function AddTable(pid As Short, tid As Byte) As Integer

	<PreserveSig> _
	Function AddExtension(pid As Short, tid As Byte, eid As Short) As Integer

	<PreserveSig> _
	Function RemovePID(pid As Short) As Integer

	<PreserveSig> _
	Function RemoveTable(pid As Short, tid As Byte) As Integer

	<PreserveSig> _
	Function RemoveExtension(pid As Short, tid As Byte, eid As Short) As Integer

End Interface

#End If

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("9B396D40-F380-4E3C-A514-1A82BF6EBFE6")> _
Public Interface IMpeg2Data
	<PreserveSig> _
	Function GetSection(<[In]> pid As Short, <[In]> tid As Byte, <[In]> pFilter As MPEG2Filter, <[In]> dwTimeout As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppSectionList As ISectionList) As Integer

	<PreserveSig> _
	Function GetTable(<[In]> pid As Short, <[In]> tid As Byte, <[In]> pFilter As MPEG2Filter, <[In]> dwTimeout As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppSectionList As ISectionList) As Integer

	<PreserveSig> _
	Function GetStreamOfSections(<[In]> pid As Short, <[In]> tid As Byte, <[In]> pFilter As MPEG2Filter, <[In]> hDataReadyEvent As IntPtr, <MarshalAs(UnmanagedType.[Interface])> ByRef ppMpegStream As IMpeg2Stream) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("400CC286-32A0-4CE4-9041-39571125A635"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IMpeg2Stream
	<PreserveSig> _
	Function Initialize(<[In]> requestType As MPEGRequestType, <[In], MarshalAs(UnmanagedType.[Interface])> pMpeg2Data As IMpeg2Data, <[In], MarshalAs(UnmanagedType.LPStruct)> pContext As MPEGContext, <[In]> pid As Short, <[In]> tid As Byte, <[In], MarshalAs(UnmanagedType.LPStruct)> pFilter As MPEG2Filter, _
		<[In]> hDataReadyEvent As IntPtr) As Integer

	<PreserveSig> _
	Function SupplyDataBuffer(<[In]> pStreamBuffer As MPEGStreamBuffer) As Integer
End Interface

<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("AFEC1EB5-2A64-46C6-BF4B-AE3CCB6AFDB0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface ISectionList
	<PreserveSig> _
	Function Initialize(<[In]> requestType As MPEGRequestType, <[In], MarshalAs(UnmanagedType.[Interface])> pMpeg2Data As IMpeg2Data, <[In], MarshalAs(UnmanagedType.LPStruct)> pContext As MPEGContext, <[In]> pid As Short, <[In]> tid As Byte, <[In], MarshalAs(UnmanagedType.LPStruct)> pFilter As MPEG2Filter, _
		<[In]> timeout As Integer, <[In]> hDoneEvent As IntPtr) As Integer

	<PreserveSig> _
	Function InitializeWithRawSections(<[In]> ByRef pmplSections As MPEGPacketList) As Integer

	<PreserveSig> _
	Function CancelPendingRequest() As Integer

	<PreserveSig> _
	Function GetNumberOfSections(ByRef pCount As Short) As Integer

    ' PSECTION*
	<PreserveSig> _
	Function GetSectionData(<[In]> SectionNumber As Short, <Out> ByRef pdwRawPacketLength As Integer, <Out> ByRef ppSection As IntPtr) As Integer

	<PreserveSig> _
	Function GetProgramIdentifier(ByRef pPid As Short) As Integer

	<PreserveSig> _
	Function GetTableIdentifier(ByRef pTableId As Byte) As Integer
End Interface

