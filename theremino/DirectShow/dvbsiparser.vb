
Imports System.Runtime.InteropServices

Namespace BDA

#If ALLOW_UNTESTED_INTERFACES Then

	Public Enum DVB_STRCONV_MODE
		DVB = 0
		DVB_EMPHASIS = (DVB + 1)
		DVB_WITHOUT_EMPHASIS = (DVB_EMPHASIS + 1)
		ISDB = (DVB_WITHOUT_EMPHASIS + 1)
	End Enum

#End If


    Public Enum RunningStatus As Byte
        Undefined = 0
        NotRunning = 1
        StartInAFewSeconds = 2
        Pausing = 3
        Running = 4
        Reserved1 = 5
        Reserved2 = 6
        Reserved3 = 7
    End Enum


#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("B758A7BD-14DC-449d-B828-35909ACB3B1E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbSiParser
		<PreserveSig> _
		Function Initialize(<[In]> punkMpeg2Data As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetPAT(<Out> ByRef ppPAT As IPAT) As Integer

		<PreserveSig> _
		Function GetCAT(<[In]> dwTimeout As Integer, <Out> ByRef ppCAT As ICAT) As Integer

		<PreserveSig> _
		Function GetPMT(<[In]> pid As Short, <[In]> pwProgramNumber As DsShort, <Out> ByRef ppPMT As IPMT) As Integer

		<PreserveSig> _
		Function GetTSDT(<Out> ByRef ppTSDT As ITSDT) As Integer

		<PreserveSig> _
		Function GetNIT(<[In]> tableId As Byte, <[In]> pwNetworkId As DsShort, <Out> ByRef ppNIT As IDVB_NIT) As Integer

		<PreserveSig> _
		Function GetSDT(<[In]> tableId As Byte, <[In]> pwTransportStreamId As DsShort, <Out> ByRef ppSDT As IDVB_SDT) As Integer

		<PreserveSig> _
		Function GetEIT(<[In]> tableId As Byte, <[In]> pwServiceId As DsShort, <Out> ByRef ppEIT As IDVB_EIT) As Integer

		<PreserveSig> _
		Function GetBAT(<[In]> pwBouquetId As DsShort, <Out> ByRef ppBAT As IDVB_BAT) As Integer

		<PreserveSig> _
		Function GetRST(<[In]> dwTimeout As Integer, <Out> ByRef ppRST As IDVB_RST) As Integer

		<PreserveSig> _
		Function GetST(<[In]> pid As Short, <[In]> dwTimeout As Integer, <Out> ByRef ppST As IDVB_ST) As Integer

		<PreserveSig> _
		Function GetTDT(<Out> ByRef ppTDT As IDVB_TDT) As Integer

		<PreserveSig> _
		Function GetTOT(<Out> ByRef ppTOT As IDVB_TOT) As Integer

		<PreserveSig> _
		Function GetDIT(<[In]> dwTimeout As Integer, <Out> ByRef ppDIT As IDVB_DIT) As Integer

		<PreserveSig> _
		Function GetSIT(<[In]> dwTimeout As Integer, <Out> ByRef ppSIT As IDVB_SIT) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("F47DCD04-1E23-4fb7-9F96-B40EEAD10B2B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDVB_RST
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordTransportStreamId(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordOriginalNetworkId(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordServiceId(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordEventId(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordRunningStatus(<[In]> dwRecordIndex As Integer, <Out> ByRef pbVal As RunningStatus) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("4D5B9F23-2A02-45de-BCDA-5D5DBFBFBE62"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDVB_ST
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList) As Integer

		<PreserveSig> _
		Function GetDataLength(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetData(<Out> ByRef ppData As IntPtr) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("0780DC7D-D55C-4aef-97E6-6B75906E2796"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDVB_TDT
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList) As Integer

		<PreserveSig> _
		Function GetUTCTime(<Out> ByRef pmdtVal As MpegDateAndTime) As Integer
	End Interface


	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("91BFFDF9-9432-410f-86EF-1C228ED0AD70"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDVB_DIT
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList) As Integer

		<PreserveSig> _
		Function GetTransitionFlag(<Out, MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("68CDCE53-8BEA-45c2-9D9D-ACF575A089B5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDVB_SIT
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList, <[In]> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfTableDescriptors(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByIndex(<[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByTag(<[In]> bTag As Byte, <[In]> pdwCookie As IntPtr, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordServiceId(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordRunningStatus(<[In]> dwRecordIndex As Integer, <Out> ByRef pbVal As RunningStatus) As Integer

		<PreserveSig> _
		Function GetRecordCountOfDescriptors(<[In]> dwRecordIndex As Integer, <Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByTag(<[In]> dwRecordIndex As Integer, <[In]> bTag As Byte, <[In], Out> pdwCookie As DsInt, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function RegisterForNextTable(<[In]> hNextTableAvailable As IntPtr) As Integer

		<PreserveSig> _
		Function GetNextTable(<[In]> dwTimeout As Integer, <Out> ByRef ppSIT As IDVB_SIT) As Integer

		<PreserveSig> _
		Function RegisterForWhenCurrent(<[In]> hNextTableIsCurrent As IntPtr) As Integer

		<PreserveSig> _
		Function ConvertNextToCurrent() As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("DFB98E36-9E1A-4862-9946-993A4E59017B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbCableDeliverySystemDescriptor
		<PreserveSig> _
		Function GetTag(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetFrequency(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetFECOuter(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetModulation(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSymbolRate(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetFECInner(<Out> ByRef pbVal As Byte) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1CADB613-E1DD-4512-AFA8-BB7A007EF8B1"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbFrequencyListDescriptor
		<PreserveSig> _
		Function GetTag(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCodingType(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCentreFrequency(<[In]> bRecordIndex As Byte, <Out> ByRef pdwVal As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("F9C7FBCF-E2D6-464d-B32D-2EF526E49290"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbServiceDescriptor
		<PreserveSig> _
		Function GetTag(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetServiceType(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetServiceProviderName(<Out, MarshalAs(UnmanagedType.LPWStr)> ByRef pszName As String) As Integer

		<PreserveSig> _
		Function GetServiceProviderNameW(<Out, MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

		<PreserveSig> _
		Function GetServiceName(<Out, MarshalAs(UnmanagedType.LPWStr)> ByRef pszName As String) As Integer

		<PreserveSig> _
		Function GetProcessedServiceName(<Out, MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

		<PreserveSig> _
		Function GetServiceNameEmphasized(<Out, MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("91E405CF-80E7-457F-9096-1B9D1CE32141"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbComponentDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetStreamContent(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetComponentType(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetComponentTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLanguageCode(ByRef pszCode As Byte) As Integer

		<PreserveSig> _
		Function GetTextW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrText As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("2E883881-A467-412A-9D63-6F2B6DA05BF0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbContentDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordContentNibbles(<[In]> bRecordIndex As Byte, ByRef pbValLevel1 As Byte, ByRef pbValLevel2 As Byte) As Integer

		<PreserveSig> _
		Function GetRecordUserNibbles(<[In]> bRecordIndex As Byte, ByRef pbVal1 As Byte, ByRef pbVal2 As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("05E0C1EA-F661-4053-9FBF-D93B28359838"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), ComConversionLoss> _
	Public Interface IDvbContentIdentifierDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCrid(<[In]> bRecordIndex As Byte, ByRef pbType As Byte, ByRef pbLocation As Byte, ByRef pbLength As Byte, <Out> ppbBytes As IntPtr) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("D1EBC1D6-8B60-4C20-9CAF-E59382E7C400"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbDataBroadcastDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetDataBroadcastID(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetComponentTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSelectorLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSelectorBytes(<[In], Out> ByRef pbLen As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLangID(ByRef pulVal As Integer) As Integer

		<PreserveSig> _
		Function GetTextLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetText(<[In], Out> ByRef pbLen As Byte, ByRef pbVal As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("5F26F518-65C8-4048-91F2-9290F59F7B90")> _
	Public Interface IDvbDataBroadcastIDDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetDataBroadcastID(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetIDSelectorBytes(<[In], Out> ByRef pbLen As Byte, ByRef pbVal As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("05EC24D1-3A31-44E7-B408-67C60A352276")> _
	Public Interface IDvbDefaultAuthorityDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetDefaultAuthority(ByRef pbLength As Byte, <Out> ppbBytes As IntPtr) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("C9B22ECA-85F4-499F-B1DB-EFA93A91EE57"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbExtendedEventDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetDescriptorNumber(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLastDescriptorNumber(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLanguageCode(ByRef pszCode As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordItemW(<[In]> bRecordIndex As Byte, <[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrDesc As String, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrItem As String) As Integer

		<PreserveSig> _
		Function GetConcatenatedItemW(<[In], MarshalAs(UnmanagedType.[Interface])> pFollowingDescriptor As IDvbExtendedEventDescriptor, <[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrDesc As String, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrItem As String) As Integer

		<PreserveSig> _
		Function GetTextW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrText As String) As Integer

		<PreserveSig> _
		Function GetConcatenatedTextW(<[In], MarshalAs(UnmanagedType.[Interface])> FollowingDescriptor As IDvbExtendedEventDescriptor, <[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrText As String) As Integer

		<PreserveSig> _
		Function GetRecordItemRawBytes(<[In]> bRecordIndex As Byte, <Out> ppbRawItem As IntPtr, ByRef pbItemLength As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("43ACA974-4BE8-4b98-BC17-9EAFD788B1D7"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbLogicalChannelDescriptor2
		Inherits IDvbLogicalChannelDescriptor

        <PreserveSig()> _
        Shadows Function GetTag(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetLength(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetCountOfRecords(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordServiceId(<[In]()> ByVal bRecordIndex As Byte, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordLogicalChannelNumber(<[In]()> ByVal bRecordIndex As Byte, <Out()> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordLogicalChannelAndVisibility(bRecordIndex As Byte, ByRef pwVal As Short) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1EA8B738-A307-4680-9E26-D0A908C824F4"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbHDSimulcastLogicalChannelDescriptor
		Inherits IDvbLogicalChannelDescriptor2

        <PreserveSig()> _
        Shadows Function GetTag(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetLength(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetCountOfRecords(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordServiceId(<[In]()> ByVal bRecordIndex As Byte, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordLogicalChannelNumber(<[In]()> ByVal bRecordIndex As Byte, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordLogicalChannelAndVisibility(ByVal bRecordIndex As Byte, ByRef pwVal As Short) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1CDF8B31-994A-46FC-ACFD-6A6BE8934DD5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbLinkageDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTSId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetONId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetServiceId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetLinkageType(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetPrivateDataLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetPrivateData(<[In], Out> ByRef pbLen As Byte, ByRef pbData As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("F69C3747-8A30-4980-998C-01FE7F0BA35A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbLogicalChannel2Descriptor
		Inherits IDvbLogicalChannelDescriptor2

        <PreserveSig()> _
        Shadows Function GetTag(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetLength(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetCountOfRecords(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordServiceId(<[In]()> ByVal bRecordIndex As Byte, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordLogicalChannelNumber(<[In]()> ByVal bRecordIndex As Byte, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordLogicalChannelAndVisibility(ByVal bRecordIndex As Byte, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetCountOfLists(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetListId(<[In]> bListIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetListNameW(<[In]> bListIndex As Byte, <[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

		<PreserveSig> _
		Function GetListCountryCode(<[In]> bListIndex As Byte, ByRef pszCode As Byte) As Integer

		<PreserveSig> _
		Function GetListCountOfRecords(<[In]> bChannelListIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetListRecordServiceId(<[In]> bListIndex As Byte, <[In]> bRecordIndex As Byte, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetListRecordLogicalChannelNumber(<[In]> bListIndex As Byte, <[In]> bRecordIndex As Byte, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetListRecordLogicalChannelAndVisibility(<[In]> bListIndex As Byte, <[In]> bRecordIndex As Byte, ByRef pwVal As Short) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("2D80433B-B32C-47EF-987F-E78EBB773E34"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbMultilingualServiceNameDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordLangId(<[In]> bRecordIndex As Byte, ByRef ulVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordServiceProviderNameW(<[In]> bRecordIndex As Byte, <[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

		<PreserveSig> _
		Function GetRecordServiceNameW(<[In]> bRecordIndex As Byte, <[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("5B2A80CF-35B9-446C-B3E4-048B761DBC51"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbNetworkNameDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetNetworkName(<Out> pszName As IntPtr) As Integer

		<PreserveSig> _
		Function GetNetworkNameW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("3AD9DDE1-FB1B-4186-937F-22E6B5A72A10")> _
	Public Interface IDvbParentalRatingDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordRating(<[In]> bRecordIndex As Byte, ByRef pszCountryCode As Byte, ByRef pbVal As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("5660A019-E75A-4B82-9B4C-ED2256D165A2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbPrivateDataSpecifierDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetPrivateDataSpecifier(ByRef pdwVal As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("0F37BD92-D6A1-4854-B950-3A969D27F30E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbServiceAttributeDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordServiceId(<[In]> bRecordIndex As Byte, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordNumericSelectionFlag(<[In]> bRecordIndex As Byte, ByRef pfVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordVisibleServiceFlag(<[In]> bRecordIndex As Byte, ByRef pfVal As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("D6C76506-85AB-487C-9B2B-36416511E4A2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbServiceDescriptor2
		Inherits IDvbServiceDescriptor

        <PreserveSig()> _
        Shadows Function GetTag(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetLength(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetServiceType(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetServiceProviderName(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef pszName As String) As Integer

        <PreserveSig()> _
        Shadows Function GetServiceProviderNameW(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

        <PreserveSig()> _
        Shadows Function GetServiceName(<Out(), MarshalAs(UnmanagedType.LPWStr)> ByRef pszName As String) As Integer

        <PreserveSig()> _
        Shadows Function GetProcessedServiceName(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

        <PreserveSig()> _
        Shadows Function GetServiceNameEmphasized(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

        <PreserveSig()> _
        Shadows Function GetServiceProviderNameW(<[In]()> ByVal convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

        <PreserveSig()> _
        Shadows Function GetServiceNameW(<[In]()> ByVal convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("05DB0D8F-6008-491A-ACD3-7090952707D0"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbServiceListDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordServiceId(<[In]> bRecordIndex As Byte, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordServiceType(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("B170BE92-5B75-458E-9C6E-B0008231491A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbShortEventDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLanguageCode(ByRef pszCode As Byte) As Integer

		<PreserveSig> _
		Function GetEventNameW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

		<PreserveSig> _
		Function GetTextW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrText As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("0AC5525F-F816-42F4-93BA-4C0F32F46E54"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbSiParser2
		Inherits IDvbSiParser

        <PreserveSig()> _
        Shadows Function Initialize(<[In]()> ByVal punkMpeg2Data As IMpeg2Data) As Integer

        <PreserveSig()> _
        Shadows Function GetPAT(<Out()> ByRef ppPAT As IPAT) As Integer

        <PreserveSig()> _
        Shadows Function GetCAT(<[In]()> ByVal dwTimeout As Integer, <Out()> ByRef ppCAT As ICAT) As Integer

        <PreserveSig()> _
        Shadows Function GetPMT(<[In]()> ByVal pid As Short, <[In]()> ByVal pwProgramNumber As DsShort, <Out()> ByRef ppPMT As IPMT) As Integer

        <PreserveSig()> _
        Shadows Function GetTSDT(<Out()> ByRef ppTSDT As ITSDT) As Integer

        <PreserveSig()> _
        Shadows Function GetNIT(<[In]()> ByVal tableId As Byte, <[In]()> ByVal pwNetworkId As DsShort, <Out()> ByRef ppNIT As IDVB_NIT) As Integer

        <PreserveSig()> _
        Shadows Function GetSDT(<[In]()> ByVal tableId As Byte, <[In]()> ByVal pwTransportStreamId As DsShort, <Out()> ByRef ppSDT As IDVB_SDT) As Integer

        <PreserveSig()> _
        Shadows Function GetEIT(<[In]()> ByVal tableId As Byte, <[In]()> ByVal pwServiceId As DsShort, <Out()> ByRef ppEIT As IDVB_EIT) As Integer

        <PreserveSig()> _
        Shadows Function GetBAT(<[In]()> ByVal pwBouquetId As DsShort, <Out()> ByRef ppBAT As IDVB_BAT) As Integer

        <PreserveSig()> _
        Shadows Function GetRST(<[In]()> ByVal dwTimeout As Integer, <Out()> ByRef ppRST As IDVB_RST) As Integer

        <PreserveSig()> _
        Shadows Function GetST(<[In]()> ByVal pid As Short, <[In]()> ByVal dwTimeout As Integer, <Out()> ByRef ppST As IDVB_ST) As Integer

        <PreserveSig()> _
        Shadows Function GetTDT(<Out()> ByRef ppTDT As IDVB_TDT) As Integer

        <PreserveSig()> _
        Shadows Function GetTOT(<Out()> ByRef ppTOT As IDVB_TOT) As Integer

        <PreserveSig()> _
        Shadows Function GetDIT(<[In]()> ByVal dwTimeout As Integer, <Out()> ByRef ppDIT As IDVB_DIT) As Integer

        <PreserveSig()> _
        Shadows Function GetSIT(<[In]()> ByVal dwTimeout As Integer, <Out()> ByRef ppSIT As IDVB_SIT) As Integer

		<PreserveSig> _
		Function GetEIT2(<[In]> TableId As Byte, <[In]> ByRef pwServiceId As Short, <[In]> ByRef pbSegment As Byte, <MarshalAs(UnmanagedType.[Interface])> ByRef ppEIT As IDVB_EIT2) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("9B25FE1D-FA23-4E50-9784-6DF8B26F8A49"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbSubtitlingDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordLangId(<[In]> bRecordIndex As Byte, ByRef pulVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordSubtitlingType(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCompositionPageID(<[In]> bRecordIndex As Byte, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordAncillaryPageID(<[In]> bRecordIndex As Byte, ByRef pwVal As Short) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("9CD29D47-69C6-4F92-98A9-210AF1B7303A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbTeletextDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordLangId(<[In]> bRecordIndex As Byte, ByRef pulVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordTeletextType(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordMagazineNumber(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordPageNumber(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("537CD71E-0E46-4173-9001-BA043F3E49E2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IISDB_BIT
		<PreserveSig> _
		Function Initialize(<[In], MarshalAs(UnmanagedType.[Interface])> pSectionList As ISectionList, <[In], MarshalAs(UnmanagedType.[Interface])> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetOriginalNetworkId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetBroadcastViewPropriety(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfTableDescriptors(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByIndex(<[In]> dwIndex As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByTag(<[In]> bTag As Byte, <[In], Out> ByRef pdwCookie As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordBroadcasterId(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCountOfDescriptors(<[In]> dwRecordIndex As Integer, ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByTag(<[In]> dwRecordIndex As Integer, <[In]> bTag As Byte, <[In], Out> ByRef pdwCookie As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetVersionHash(ByRef pdwVersionHash As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("25FA92C2-8B80-4787-A841-3A0E8F17984B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IISDB_CDT
		<PreserveSig> _
		Function Initialize(<[In], MarshalAs(UnmanagedType.[Interface])> pSectionList As ISectionList, <[In], MarshalAs(UnmanagedType.[Interface])> pMPEGData As IMpeg2Data, <[In]> bSectionNumber As Byte) As Integer

		<PreserveSig> _
		Function GetVersionNumber(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetDownloadDataId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetSectionNumber(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetOriginalNetworkId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetDataType(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfTableDescriptors(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByIndex(<[In]> dwIndex As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByTag(<[In]> bTag As Byte, <[In], Out> ByRef pdwCookie As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetSizeOfDataModule(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetDataModule(<Out> pbData As IntPtr) As Integer

		<PreserveSig> _
		Function GetVersionHash(ByRef pdwVersionHash As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("0EDB556D-43AD-4938-9668-321B2FFECFD3"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IISDB_EMM
		<PreserveSig> _
		Function Initialize(<[In], MarshalAs(UnmanagedType.[Interface])> pSectionList As ISectionList, <[In], MarshalAs(UnmanagedType.[Interface])> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTableIdExtension(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetDataBytes(<[In], Out> ByRef pwBufferLength As Short, ByRef pbBuffer As Byte) As Integer

		<PreserveSig> _
		Function GetSharedEmmMessage(ByRef pwLength As Short, ppbMessage As IntPtr) As Integer

		<PreserveSig> _
		Function GetIndividualEmmMessage(<MarshalAs(UnmanagedType.IUnknown)> pUnknown As Object, ByRef pwLength As Short, ppbMessage As IntPtr) As Integer

		<PreserveSig> _
		Function GetVersionHash(ByRef pdwVersionHash As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("141A546B-02FF-4FB9-A3A3-2F074B74A9A9"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IISDB_LDT
		<PreserveSig> _
		Function Initialize(<[In], MarshalAs(UnmanagedType.[Interface])> pSectionList As ISectionList, <[In], MarshalAs(UnmanagedType.[Interface])> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetOriginalServiceId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetTransportStreamId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetOriginalNetworkId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordDescriptionId(<[In]> dwRecordIndex As Integer, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordCountOfDescriptors(<[In]> dwRecordIndex As Integer, ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByTag(<[In]> dwRecordIndex As Integer, <[In]> bTag As Byte, <[In], Out> ByRef pdwCookie As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetVersionHash(ByRef pdwVersionHash As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1B1863EF-08F1-40B7-A559-3B1EFF8CAFA6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IISDB_NBIT
		<PreserveSig> _
		Function Initialize(<[In], MarshalAs(UnmanagedType.[Interface])> pSectionList As ISectionList, <[In], MarshalAs(UnmanagedType.[Interface])> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetOriginalNetworkId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordInformationId(<[In]> dwRecordIndex As Integer, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordInformationType(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordDescriptionBodyLocation(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordMessageSectionNumber(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordUserDefined(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordNumberOfKeys(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordKeys(<[In]> dwRecordIndex As Integer, <Out> pbKeys As IntPtr) As Integer

		<PreserveSig> _
		Function GetRecordCountOfDescriptors(<[In]> dwRecordIndex As Integer, ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByTag(<[In]> dwRecordIndex As Integer, <[In]> bTag As Byte, <[In], Out> ByRef pdwCookie As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetVersionHash(ByRef pdwVersionHash As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("3F3DC9A2-BB32-4FB9-AE9E-D856848927A3"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IISDB_SDT
		Inherits IDVB_SDT

        <PreserveSig()> _
        Shadows Function Initialize(<[In]()> ByVal pSectionList As ISectionList, <[In]()> ByVal pMPEGData As IMpeg2Data) As Integer

        <PreserveSig()> _
        Shadows Function GetVersionNumber(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Shadows Function GetTransportStreamId(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Shadows Function GetOriginalNetworkId(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Shadows Function GetCountOfRecords(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordServiceId(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordEITScheduleFlag(<[In]()> ByVal dwRecordIndex As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordEITPresentFollowingFlag(<[In]()> ByVal dwRecordIndex As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordRunningStatus(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pbVal As RunningStatus) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordFreeCAMode(<[In]()> ByVal dwRecordIndex As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordCountOfDescriptors(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordDescriptorByIndex(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal dwIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Shadows Function GetRecordDescriptorByTag(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal bTag As Byte, <[In](), Out()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Shadows Function RegisterForNextTable(<[In]()> ByVal hNextTableAvailable As IntPtr) As Integer

        <PreserveSig()> _
        Shadows Function GetNextTable(<Out()> ByRef ppSDT As IDVB_SDT) As Integer

        <PreserveSig()> _
        Shadows Function RegisterForWhenCurrent(<[In]()> ByVal hNextTableIsCurrent As IntPtr) As Integer

        <PreserveSig()> _
        Shadows Function ConvertNextToCurrent() As Integer

        <PreserveSig()> _
        Shadows Function GetVersionHash(<Out()> ByRef pdwVersionHash As Integer) As Integer

		<PreserveSig> _
		Function GetRecordEITUserDefinedFlags(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("EE60EF2D-813A-4DC7-BF92-EA13DAC85313"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IISDB_SDTT
		<PreserveSig> _
		Function Initialize(<[In], MarshalAs(UnmanagedType.[Interface])> pSectionList As ISectionList, <[In], MarshalAs(UnmanagedType.[Interface])> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTableIdExt(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetTransportStreamId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetOriginalNetworkId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetServiceId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordGroup(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordTargetVersion(<[In]> dwRecordIndex As Integer, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordNewVersion(<[In]> dwRecordIndex As Integer, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordDownloadLevel(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordVersionIndicator(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordScheduleTimeShiftInformation(<[In]> dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCountOfSchedules(<[In]> dwRecordIndex As Integer, ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordStartTimeByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, ByRef pmdtVal As MpegDateAndTime) As Integer

		<PreserveSig> _
		Function GetRecordDurationByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, ByRef pmdVal As MpegDuration) As Integer

		<PreserveSig> _
		Function GetRecordCountOfDescriptors(<[In]> dwRecordIndex As Integer, ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByTag(<[In]> dwRecordIndex As Integer, <[In]> bTag As Byte, <[In], Out> ByRef pdwCookie As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetVersionHash(ByRef pdwVersionHash As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("679D2002-2425-4BE4-A4C7-D6632A574F4D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbAudioComponentDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetStreamContent(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetComponentType(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetComponentTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetStreamType(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSimulcastGroupTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetESMultiLingualFlag(ByRef pfVal As Integer) As Integer

		<PreserveSig> _
		Function GetMainComponentFlag(ByRef pfVal As Integer) As Integer

		<PreserveSig> _
		Function GetQualityIndicator(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSamplingRate(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLanguageCode(ByRef pszCode As Byte) As Integer

		<PreserveSig> _
		Function GetLanguageCode2(ByRef pszCode As Byte) As Integer

		<PreserveSig> _
		Function GetTextW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrText As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("08E18B25-A28F-4E92-821E-4FCED5CC2291"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbCAContractInformationDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCASystemId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetCAUnitId(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordComponentTag(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetContractVerificationInfoLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetContractVerificationInfo(<[In]> bBufLength As Byte, ByRef pbBuf As Byte) As Integer

		<PreserveSig> _
		Function GetFeeNameW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("0570AA47-52BC-42AE-8CA5-969F41E81AEA"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbCADescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCASystemId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetReservedBits(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCAPID(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetPrivateDataBytes(<[In], Out> ByRef pbBufferLength As Byte, ByRef pbBuffer As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("39CBEB97-FF0B-42A7-9AB9-7B9CFE70A77A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbCAServiceDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCASystemId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetCABroadcasterGroupId(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetMessageControl(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetServiceIds(<[In], Out> ByRef pbNumServiceIds As Byte, ByRef pwServiceIds As Short) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("A494F17F-C592-47D8-8943-64C9A34BE7B9"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbComponentGroupDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetComponentGroupType(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordGroupId(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordNumberOfCAUnit(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCAUnitCAUnitId(<[In]> bRecordIndex As Byte, <[In]> bCAUnitIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCAUnitNumberOfComponents(<[In]> bRecordIndex As Byte, <[In]> bCAUnitIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCAUnitComponentTag(<[In]> bRecordIndex As Byte, <[In]> bCAUnitIndex As Byte, <[In]> bComponentIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordTotalBitRate(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordTextW(<[In]> bRecordIndex As Byte, <[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrText As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("A428100A-E646-4BD6-AA14-6087BDC08CD5"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbDataContentDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetDataComponentId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetEntryComponent(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSelectorLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSelectorBytes(<[In]> bBufLength As Byte, ByRef pbBuf As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordComponentRef(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLanguageCode(ByRef pszCode As Byte) As Integer

		<PreserveSig> _
		Function GetTextW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrText As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1A28417E-266A-4BB8-A4BD-D782BCFB8161"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbDigitalCopyControlDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCopyControl(ByRef pbDigitalRecordingControlData As Byte, ByRef pbCopyControlType As Byte, ByRef pbAPSControlData As Byte, ByRef pbMaximumBitrate As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCopyControl(<[In]> bRecordIndex As Byte, ByRef pbComponentTag As Byte, ByRef pbDigitalRecordingControlData As Byte, ByRef pbCopyControlType As Byte, ByRef pbAPSControlData As Byte, ByRef pbMaximumBitrate As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("5298661E-CB88-4F5F-A1DE-5F440C185B92"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbDownloadContentDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetFlags(ByRef pfReboot As Integer, ByRef pfAddOn As Integer, ByRef pfCompatibility As Integer, ByRef pfModuleInfo As Integer, ByRef pfTextInfo As Integer) As Integer

		<PreserveSig> _
		Function GetComponentSize(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetDownloadId(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetTimeOutValueDII(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetLeakRate(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetComponentTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCompatiblityDescriptorLength(ByRef pwLength As Short) As Integer

		<PreserveSig> _
		Function GetCompatiblityDescriptor(<Out> ppbData As IntPtr) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordModuleId(<[In]> wRecordIndex As Short, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordModuleSize(<[In]> wRecordIndex As Short, ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordModuleInfoLength(<[In]> wRecordIndex As Short, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordModuleInfo(<[In]> wRecordIndex As Short, <Out> ppbData As IntPtr) As Integer

		<PreserveSig> _
		Function GetTextLanguageCode(ByRef szCode As Byte) As Integer

		<PreserveSig> _
		Function GetTextW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("BA6FA681-B973-4DA1-9207-AC3E7F0341EB"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbEmergencyInformationDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetServiceId(<[In]> bRecordIndex As Byte, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetStartEndFlag(<[In]> bRecordIndex As Byte, ByRef pVal As Byte) As Integer

		<PreserveSig> _
		Function GetSignalLevel(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetAreaCode(<[In]> bRecordIndex As Byte, <Out> ppwVal As IntPtr, ByRef pbNumAreaCodes As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("94B06780-2E2A-44DC-A966-CC56FDABC6C2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbEventGroupDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetGroupType(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordEvent(<[In]> bRecordIndex As Byte, ByRef pwServiceId As Short, ByRef pwEventId As Short) As Integer

		<PreserveSig> _
		Function GetCountOfRefRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRefRecordEvent(<[In]> bRecordIndex As Byte, ByRef pwOriginalNetworkId As Short, ByRef pwTransportStreamId As Short, ByRef pwServiceId As Short, ByRef pwEventId As Short) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("B7B3AE90-EE0B-446D-8769-F7E2AA266AA6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbHierarchicalTransmissionDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetFutureUse1(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetQualityLevel(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetFutureUse2(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetReferencePid(ByRef pwVal As Short) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("E0103F49-4AE1-4F07-9098-756DB1FA88CD"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbLogoTransmissionDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLogoTransmissionType(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLogoId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetLogoVersion(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetDownloadDataId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetLogoCharW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrChar As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("07EF6370-1660-4F26-87FC-614ADAB24B11"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbSeriesDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSeriesId(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRepeatLabel(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetProgramPattern(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetExpireDate(ByRef pfValid As Integer, ByRef pmdtVal As MpegDateAndTime) As Integer

		<PreserveSig> _
		Function GetEpisodeNumber(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetLastEpisodeNumber(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetSeriesNameW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("F837DC36-867C-426A-9111-F62093951A45"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbSIParameterDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetParameterVersion(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetUpdateTime(ByRef pVal As MpegDate) As Integer

		<PreserveSig> _
		Function GetRecordNumberOfTable(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTableId(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTableDescriptionLength(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTableDescriptionBytes(<[In]> bRecordIndex As Byte, <[In], Out> ByRef pbBufferLength As Byte, ByRef pbBuffer As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("900E4BB7-18CD-453F-98BE-3BE6AA211772"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbSiParser2
		Inherits IDvbSiParser2

		<PreserveSig> _
		Shadows Function Initialize(<[In]> punkMpeg2Data As IMpeg2Data) As Integer

		<PreserveSig> _
		Shadows Function GetPAT(<Out> ByRef ppPAT As IPAT) As Integer

		<PreserveSig> _
		Shadows Function GetCAT(<[In]> dwTimeout As Integer, <Out> ByRef ppCAT As ICAT) As Integer

		<PreserveSig> _
		Shadows Function GetPMT(<[In]> pid As Short, <[In]> pwProgramNumber As DsShort, <Out> ByRef ppPMT As IPMT) As Integer

		<PreserveSig> _
		Shadows Function GetTSDT(<Out> ByRef ppTSDT As ITSDT) As Integer

		<PreserveSig> _
		Shadows Function GetNIT(<[In]> tableId As Byte, <[In]> pwNetworkId As DsShort, <Out> ByRef ppNIT As IDVB_NIT) As Integer

		<PreserveSig> _
		Shadows Function GetSDT(<[In]> tableId As Byte, <[In]> pwTransportStreamId As DsShort, <Out> ByRef ppSDT As IDVB_SDT) As Integer

		<PreserveSig> _
		Shadows Function GetEIT(<[In]> tableId As Byte, <[In]> pwServiceId As DsShort, <Out> ByRef ppEIT As IDVB_EIT) As Integer

		<PreserveSig> _
		Shadows Function GetBAT(<[In]> pwBouquetId As DsShort, <Out> ByRef ppBAT As IDVB_BAT) As Integer

		<PreserveSig> _
		Shadows Function GetRST(<[In]> dwTimeout As Integer, <Out> ByRef ppRST As IDVB_RST) As Integer

		<PreserveSig> _
		Shadows Function GetST(<[In]> pid As Short, <[In]> dwTimeout As Integer, <Out> ByRef ppST As IDVB_ST) As Integer

		<PreserveSig> _
		Shadows Function GetTDT(<Out> ByRef ppTDT As IDVB_TDT) As Integer

		<PreserveSig> _
		Shadows Function GetTOT(<Out> ByRef ppTOT As IDVB_TOT) As Integer

		<PreserveSig> _
		Shadows Function GetDIT(<[In]> dwTimeout As Integer, <Out> ByRef ppDIT As IDVB_DIT) As Integer

		<PreserveSig> _
		Shadows Function GetSIT(<[In]> dwTimeout As Integer, <Out> ByRef ppSIT As IDVB_SIT) As Integer

		<PreserveSig> _
		Shadows Function GetEIT2(<[In]> TableId As Byte, <[In]> ByRef pwServiceId As Short, <[In]> ByRef pbSegment As Byte, <MarshalAs(UnmanagedType.[Interface])> ByRef ppEIT As IDVB_EIT2) As Integer

        <PreserveSig()> _
        Shadows Function GetSDT(<[In]()> ByVal TableId As Byte, <[In]()> ByRef pwTransportStreamId As Short, <MarshalAs(UnmanagedType.[Interface])> ByRef ppSDT As IISDB_SDT) As Integer

		<PreserveSig> _
		Function GetBIT(<[In]> TableId As Byte, <[In]> ByRef pwOriginalNetworkId As Short, <MarshalAs(UnmanagedType.[Interface])> ByRef ppBIT As IISDB_BIT) As Integer

		<PreserveSig> _
		Function GetNBIT(<[In]> TableId As Byte, <[In]> ByRef pwOriginalNetworkId As Short, <MarshalAs(UnmanagedType.[Interface])> ByRef ppNBIT As IISDB_NBIT) As Integer

		<PreserveSig> _
		Function GetLDT(<[In]> TableId As Byte, <[In]> ByRef pwOriginalServiceId As Short, <MarshalAs(UnmanagedType.[Interface])> ByRef ppLDT As IISDB_LDT) As Integer

		<PreserveSig> _
		Function GetSDTT(<[In]> TableId As Byte, <[In]> ByRef pwTableIdExt As Short, <MarshalAs(UnmanagedType.[Interface])> ByRef ppSDTT As IISDB_SDTT) As Integer

		<PreserveSig> _
		Function GetCDT(<[In]> TableId As Byte, <[In]> bSectionNumber As Byte, <[In]> ByRef pwDownloadDataId As Short, <MarshalAs(UnmanagedType.[Interface])> ByRef ppCDT As IISDB_CDT) As Integer

		<PreserveSig> _
		Function GetEMM(<[In]> Pid As Short, <[In]> wTableIdExt As Short, <MarshalAs(UnmanagedType.[Interface])> ByRef ppEMM As IISDB_EMM) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("39FAE0A6-D151-44DD-A28A-765DE5991670"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbTerrestrialDeliverySystemDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetAreaCode(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetGuardInterval(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTransmissionMode(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordFrequency(<[In]> bRecordIndex As Byte, ByRef pdwVal As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("D7AD183E-38F5-4210-B55F-EC8D601BBD47"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IIsdbTSInformationDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRemoteControlKeyId(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTSNameW(<[In]> convMode As DVB_STRCONV_MODE, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrName As String) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordTransmissionTypeInfo(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordNumberOfServices(<[In]> bRecordIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordServiceIdByIndex(<[In]> bRecordIndex As Byte, <[In]> bServiceIndex As Byte, ByRef pdwVal As Short) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("A35F2DEA-098F-4EBD-984C-2BD4C3C8CE0A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IPBDA_EIT
		<PreserveSig> _
		Function Initialize(<[In]> size As Integer, <[In]> ByRef pBuffer As Byte) As Integer

		<PreserveSig> _
		Function GetTableId(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetVersionNumber(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetServiceIdx(ByRef plwVal As Long) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordEventId(<[In]> dwRecordIndex As Integer, ByRef plwVal As Long) As Integer

		<PreserveSig> _
		Function GetRecordStartTime(<[In]> dwRecordIndex As Integer, ByRef pmdtVal As MpegDateAndTime) As Integer

		<PreserveSig> _
		Function GetRecordDuration(<[In]> dwRecordIndex As Integer, ByRef pmdVal As MpegDuration) As Integer

		<PreserveSig> _
		Function GetRecordCountOfDescriptors(<[In]> dwRecordIndex As Integer, ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByTag(<[In]> dwRecordIndex As Integer, <[In]> bTag As Byte, <[In], Out> ByRef pdwCookie As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppDescriptor As IGenericDescriptor) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("944EAB37-EED4-4850-AFD2-77E7EFEB4427"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IPBDA_Services
		<PreserveSig> _
		Function Initialize(<[In]> size As Integer, <[In]> ByRef pBuffer As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordByIndex(<[In]> dwRecordIndex As Integer, ByRef pul64ServiceIdx As Long) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("313B3620-3263-45A6-9533-968BEFBEAC03"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IPBDAAttributesDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetAttributePayload(<Out> ppbAttributeBuffer As IntPtr, ByRef pdwAttributeLength As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("22632497-0DE3-4587-AADC-D8D99017E760"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IPBDAEntitlementDescriptor
		<PreserveSig> _
		Function GetTag(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetToken(<Out> ppbTokenBuffer As IntPtr, ByRef pdwTokenLength As Integer) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("9DE49A74-ABA2-4A18-93E1-21F17F95C3C3"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IPBDASiParser
		<PreserveSig> _
		Function Initialize(<[In], MarshalAs(UnmanagedType.IUnknown)> punk As Object) As Integer

		<PreserveSig> _
		Function GetEIT(<[In]> dwSize As Integer, <[In]> ByRef pBuffer As Byte, <MarshalAs(UnmanagedType.[Interface])> ByRef ppEIT As IPBDA_EIT) As Integer

		<PreserveSig> _
		Function GetServices(<[In]> dwSize As Integer, <[In]> ByRef pBuffer As Byte, <MarshalAs(UnmanagedType.[Interface])> ByRef ppServices As IPBDA_Services) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("20EE9BE9-CD57-49ab-8F6E-1D07AEB8E482"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvbTerrestrial2DeliverySystemDescriptor
		<PreserveSig> _
		Function GetTag(<Out> pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(<Out> pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTagExtension(<Out> pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCentreFrequency(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetPLPId(<Out> pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetT2SystemId(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetMultipleInputMode(<Out> pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetBandwidth(<Out> pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetGuardInterval(<Out> pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTransmissionMode(<Out> pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCellId(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetOtherFrequencyFlag(<Out> pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTFSFlag(<Out> pbVal As Byte) As Integer

	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("61A389E0-9B9E-4ba0-AEEA-5DDD159820EA"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDVB_EIT2
		Inherits IDVB_EIT

		<PreserveSig> _
		Shadows Function Initialize(<[In]> pSectionList As ISectionList, <[In]> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Shadows Function GetVersionNumber(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Shadows Function GetServiceId(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Shadows Function GetTransportStreamId(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Shadows Function GetOriginalNetworkId(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Shadows Function GetSegmentLastSectionNumber(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Shadows Function GetLastTableId(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Shadows Function GetCountOfRecords(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Shadows Function GetRecordEventId(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Shadows Function GetRecordStartTime(<[In]> dwRecordIndex As Integer, <Out> ByRef pmdtVal As MpegDateAndTime) As Integer

		<PreserveSig> _
		Shadows Function GetRecordDuration(<[In]> dwRecordIndex As Integer, <Out> ByRef pmdVal As MpegDuration) As Integer

		<PreserveSig> _
		Shadows Function GetRecordRunningStatus(<[In]> dwRecordIndex As Integer, <Out> ByRef pbVal As RunningStatus) As Integer

		<PreserveSig> _
		Shadows Function GetRecordFreeCAMode(<[In]> dwRecordIndex As Integer, <Out, MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

		<PreserveSig> _
		Shadows Function GetRecordCountOfDescriptors(<[In]> dwRecordIndex As Integer, <Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Shadows Function GetRecordDescriptorByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Shadows Function GetRecordDescriptorByTag(<[In]> dwRecordIndex As Integer, <[In]> bTag As Byte, <[In], Out> pdwCookie As DsInt, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Shadows Function RegisterForNextTable(<[In]> hNextTableAvailable As IntPtr) As Integer

		<PreserveSig> _
		Shadows Function GetNextTable(<Out> ByRef ppEIT As IDVB_EIT) As Integer

		<PreserveSig> _
		Shadows Function RegisterForWhenCurrent(<[In]> hNextTableIsCurrent As IntPtr) As Integer

		<PreserveSig> _
		Shadows Function ConvertNextToCurrent() As Integer

		<PreserveSig> _
		Shadows Function GetVersionHash(<Out> ByRef pdwVersionHash As Integer) As Integer

		<PreserveSig> _
		Function GetSegmentInfo(ByRef pbTid As Byte, ByRef pbSegment As Byte) As Integer

		<PreserveSig> _
		Function GetRecordSection(dwRecordIndex As Integer, ByRef pbVal As Byte) As Integer

	End Interface

#End If

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C64935F4-29E4-4e22-911A-63F7F55CB097"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDVB_NIT
        <PreserveSig()> _
        Function Initialize(<[In]()> ByVal pSectionList As ISectionList, <[In]()> ByVal pMPEGData As IMpeg2Data) As Integer

        <PreserveSig()> _
        Function GetVersionNumber(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetNetworkId(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetCountOfTableDescriptors(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetTableDescriptorByIndex(<[In]()> ByVal dwIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetTableDescriptorByTag(<[In]()> ByVal bTag As Byte, <[In]()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetCountOfRecords(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetRecordTransportStreamId(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetRecordOriginalNetworkId(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetRecordCountOfDescriptors(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetRecordDescriptorByIndex(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal dwIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetRecordDescriptorByTag(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal bTag As Byte, <[In](), Out()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function RegisterForNextTable(<[In]()> ByVal hNextTableAvailable As IntPtr) As Integer

        <PreserveSig()> _
        Function GetNextTable(<Out()> ByRef ppNIT As IDVB_NIT) As Integer

        <PreserveSig()> _
        Function RegisterForWhenCurrent(<[In]()> ByVal hNextTableIsCurrent As IntPtr) As Integer

        <PreserveSig()> _
        Function ConvertNextToCurrent() As Integer

        <PreserveSig()> _
        Function GetVersionHash(<Out()> ByRef pdwVersionHash As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("ECE9BB0C-43B6-4558-A0EC-1812C34CD6CA"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDVB_BAT
        <PreserveSig()> _
        Function Initialize(<[In]()> ByVal pSectionList As ISectionList, <[In]()> ByVal pMPEGData As IMpeg2Data) As Integer

        <PreserveSig()> _
        Function GetVersionNumber(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetBouquetId(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetCountOfTableDescriptors(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetTableDescriptorByIndex(<[In]()> ByVal dwIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetTableDescriptorByTag(<[In]()> ByVal bTag As Byte, <[In]()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetCountOfRecords(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetRecordTransportStreamId(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetRecordOriginalNetworkId(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetRecordCountOfDescriptors(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetRecordDescriptorByIndex(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal dwIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetRecordDescriptorByTag(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal bTag As Byte, <[In](), Out()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function RegisterForNextTable(<[In]()> ByVal hNextTableAvailable As IntPtr) As Integer

        <PreserveSig()> _
        Function GetNextTable(<Out()> ByRef ppBAT As IDVB_BAT) As Integer

        <PreserveSig()> _
        Function RegisterForWhenCurrent(<[In]()> ByVal hNextTableIsCurrent As IntPtr) As Integer

        <PreserveSig()> _
        Function ConvertNextToCurrent() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("ED7E1B91-D12E-420c-B41D-A49D84FE1823"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDvbTerrestrialDeliverySystemDescriptor
        <PreserveSig()> _
        Function GetTag(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetLength(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetCentreFrequency(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetBandwidth(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetConstellation(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetHierarchyInformation(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetCodeRateHPStream(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetCodeRateLPStream(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetGuardInterval(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetTransmissionMode(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetOtherFrequencyFlag(<Out()> ByRef pbVal As Byte) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("442DB029-02CB-4495-8B92-1C13375BCE99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDVB_EIT
        <PreserveSig()> _
        Function Initialize(<[In]()> ByVal pSectionList As ISectionList, <[In]()> ByVal pMPEGData As IMpeg2Data) As Integer

        <PreserveSig()> _
        Function GetVersionNumber(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetServiceId(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetTransportStreamId(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetOriginalNetworkId(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetSegmentLastSectionNumber(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetLastTableId(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetCountOfRecords(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetRecordEventId(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetRecordStartTime(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pmdtVal As MpegDateAndTime) As Integer

        <PreserveSig()> _
        Function GetRecordDuration(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pmdVal As MpegDuration) As Integer

        <PreserveSig()> _
        Function GetRecordRunningStatus(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pbVal As RunningStatus) As Integer

        <PreserveSig()> _
        Function GetRecordFreeCAMode(<[In]()> ByVal dwRecordIndex As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

        <PreserveSig()> _
        Function GetRecordCountOfDescriptors(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetRecordDescriptorByIndex(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal dwIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetRecordDescriptorByTag(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal bTag As Byte, <[In](), Out()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function RegisterForNextTable(<[In]()> ByVal hNextTableAvailable As IntPtr) As Integer

        <PreserveSig()> _
        Function GetNextTable(<Out()> ByRef ppEIT As IDVB_EIT) As Integer

        <PreserveSig()> _
        Function RegisterForWhenCurrent(<[In]()> ByVal hNextTableIsCurrent As IntPtr) As Integer

        <PreserveSig()> _
        Function ConvertNextToCurrent() As Integer

        <PreserveSig()> _
        Function GetVersionHash(<Out()> ByRef pdwVersionHash As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("02F2225A-805B-4ec5-A9A6-F9B5913CD470"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDvbSatelliteDeliverySystemDescriptor
        <PreserveSig()> _
        Function GetTag(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetLength(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetFrequency(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetOrbitalPosition(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetWestEastFlag(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetPolarization(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetModulation(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetSymbolRate(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetFECInner(<Out()> ByRef pbVal As Byte) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("02CAD8D3-FE43-48e2-90BD-450ED9A8A5FD"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDVB_SDT
        <PreserveSig()> _
        Function Initialize(<[In]()> ByVal pSectionList As ISectionList, <[In]()> ByVal pMPEGData As IMpeg2Data) As Integer

        <PreserveSig()> _
        Function GetVersionNumber(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetTransportStreamId(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetOriginalNetworkId(<Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetCountOfRecords(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetRecordServiceId(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetRecordEITScheduleFlag(<[In]()> ByVal dwRecordIndex As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

        <PreserveSig()> _
        Function GetRecordEITPresentFollowingFlag(<[In]()> ByVal dwRecordIndex As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

        <PreserveSig()> _
        Function GetRecordRunningStatus(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pbVal As RunningStatus) As Integer

        <PreserveSig()> _
        Function GetRecordFreeCAMode(<[In]()> ByVal dwRecordIndex As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

        <PreserveSig()> _
        Function GetRecordCountOfDescriptors(<[In]()> ByVal dwRecordIndex As Integer, <Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetRecordDescriptorByIndex(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal dwIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetRecordDescriptorByTag(<[In]()> ByVal dwRecordIndex As Integer, <[In]()> ByVal bTag As Byte, <[In](), Out()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function RegisterForNextTable(<[In]()> ByVal hNextTableAvailable As IntPtr) As Integer

        <PreserveSig()> _
        Function GetNextTable(<Out()> ByRef ppSDT As IDVB_SDT) As Integer

        <PreserveSig()> _
        Function RegisterForWhenCurrent(<[In]()> ByVal hNextTableIsCurrent As IntPtr) As Integer

        <PreserveSig()> _
        Function ConvertNextToCurrent() As Integer

        <PreserveSig()> _
        Function GetVersionHash(<Out()> ByRef pdwVersionHash As Integer) As Integer

    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("CF1EDAFF-3FFD-4cf7-8201-35756ACBF85F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDvbLogicalChannelDescriptor
        <PreserveSig()> _
        Function GetTag(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetLength(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetCountOfRecords(<Out()> ByRef pbVal As Byte) As Integer

        <PreserveSig()> _
        Function GetRecordServiceId(<[In]()> ByVal bRecordIndex As Byte, <Out()> ByRef pwVal As Short) As Integer

        <PreserveSig()> _
        Function GetRecordLogicalChannelNumber(<[In]()> ByVal bRecordIndex As Byte, <Out()> ByRef pwVal As Short) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("83295D6A-FABA-4ee1-9B15-8067696910AE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IDVB_TOT
        <PreserveSig()> _
        Function Initialize(<[In]()> ByVal pSectionList As ISectionList) As Integer

        <PreserveSig()> _
        Function GetUTCTime(<Out()> ByRef pmdtVal As MpegDateAndTime) As Integer

        <PreserveSig()> _
        Function GetCountOfTableDescriptors(<Out()> ByRef pdwVal As Integer) As Integer

        <PreserveSig()> _
        Function GetTableDescriptorByIndex(<[In]()> ByVal dwIndex As Integer, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer

        <PreserveSig()> _
        Function GetTableDescriptorByTag(<[In]()> ByVal bTag As Byte, <[In](), Out()> ByVal pdwCookie As DsInt, <Out()> ByRef ppDescriptor As IGenericDescriptor) As Integer
    End Interface

End Namespace
