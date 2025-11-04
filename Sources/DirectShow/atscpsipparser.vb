
Imports System.Runtime.InteropServices

Namespace BDA

#If ALLOW_UNTESTED_INTERFACES Then

	Public Enum AtscEtmLocation
		NotPresent = &H0
		InPtcForPsip = &H1
		InPtcForEvent = &H2
		Reserved = &H3
	End Enum

#End If


#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("B2C98995-5EB2-4fb1-B406-F3E8E2026A9A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IAtscPsipParser
		<PreserveSig> _
		Function Initialize(<[In]> punkMpeg2Data As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetPAT(<Out> ByRef ppPAT As IPAT) As Integer

		<PreserveSig> _
		Function GetCAT(<[In]> dwTimeout As Integer, <Out> ByRef ppCAT As ICAT) As Integer

		<PreserveSig> _
		Function GetPMT(<[In]> pid As Short, <[In]> pwProgramNumber As IntPtr, <Out> ByRef ppPMT As IPMT) As Integer

		<PreserveSig> _
		Function GetTSDT(<Out> ByRef ppTSDT As ITSDT) As Integer

		<PreserveSig> _
		Function GetMGT(<Out> ByRef ppMGT As IATSC_MGT) As Integer

		<PreserveSig> _
		Function GetVCT(<[In]> tableId As Byte, <[In], MarshalAs(UnmanagedType.Bool)> fGetNextTable As Boolean, <Out> ByRef ppVCT As IATSC_VCT) As Integer

		<PreserveSig> _
		Function GetEIT(<[In]> pid As Short, <[In]> pwSourceId As IntPtr, <[In]> dwTimeout As Integer, <Out> ByRef ppEIT As IATSC_EIT) As Integer

		<PreserveSig> _
		Function GetETT(<[In]> pid As Short, <[In]> wSourceId As IntPtr, <[In]> pwEventId As IntPtr, <Out> ByRef ppETT As IATSC_ETT) As Integer

		<PreserveSig> _
		Function GetSTT(<Out> ByRef ppSTT As IATSC_STT) As Integer

		<PreserveSig> _
		Function GetEAS(<[In]> pid As Short, <Out> ByRef ppEAS As ISCTE_EAS) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("8877dabd-c137-4073-97e3-779407a5d87a"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IATSC_MGT
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList, <[In]> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetProtocolVersion(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordType(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordTypePid(<[In]> dwRecordIndex As Integer, <Out> ByRef ppidVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordVersionNumber(<[In]> dwRecordIndex As Integer, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCountOfDescriptors(<[In]> dwRecordIndex As Integer, <Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByTag(<[In]> dwRecordIndex As Integer, <[In]> bTag As Byte, <[In], Out> pdwCookie As DsInt, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetCountOfTableDescriptors(<[In], Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByIndex(<[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByTag(<[In]> bTag As Byte, <[In]> pdwCookie As IntPtr, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("26879a18-32f9-46c6-91f0-fb6479270e8c"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IATSC_VCT
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList, <[In]> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetTransportStreamId(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetProtocolVersion(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordName(<[In]> dwRecordIndex As Integer, <Out, MarshalAs(UnmanagedType.LPWStr)> ByRef pwsName As String) As Integer

		<PreserveSig> _
		Function GetRecordMajorChannelNumber(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordMinorChannelNumber(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordModulationMode(<[In]> dwRecordIndex As Integer, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordCarrierFrequency(<[In]> dwRecordIndex As Integer, <Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordTransportStreamId(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordProgramNumber(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordEtmLocation(<[In]> dwRecordIndex As Integer, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordIsAccessControlledBitSet(<[In]> dwRecordIndex As Integer, <Out, MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

		<PreserveSig> _
		Function GetRecordIsHiddenBitSet(<[In]> dwRecordIndex As Integer, <Out, MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

		<PreserveSig> _
		Function GetRecordIsPathSelectBitSet(<[In]> dwRecordIndex As Integer, <Out, MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

		<PreserveSig> _
		Function GetRecordIsOutOfBandBitSet(<[In]> dwRecordIndex As Integer, <Out, MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

		<PreserveSig> _
		Function GetRecordIsHideGuideBitSet(<[In]> dwRecordIndex As Integer, <Out, MarshalAs(UnmanagedType.Bool)> ByRef pfVal As Boolean) As Integer

		<PreserveSig> _
		Function GetRecordServiceType(<[In]> dwRecordIndex As Integer, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordSourceId(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordCountOfDescriptors(<[In]> dwRecordIndex As Integer, <Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByTag(<[In]> dwRecordIndex As Integer, <[In]> bTag As Byte, <[In], Out> pdwCookie As DsInt, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetCountOfTableDescriptors(<[In], Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByIndex(<[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByTag(<[In]> bTag As Byte, <[In]> pdwCookie As IntPtr, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("d7c212d7-76a2-4b4b-aa56-846879a80096"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IATSC_EIT
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList, <[In]> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSourceId(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetProtocolVersion(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCountOfRecords(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordEventId(<[In]> dwRecordIndex As Integer, <Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRecordStartTime(<[In]> dwRecordIndex As Integer, <Out> ByRef pmdtVal As MpegDateAndTime) As Integer

		<PreserveSig> _
		Function GetRecordEtmLocation(<[In]> dwRecordIndex As Integer, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordDuration(<[In]> dwRecordIndex As Integer, <Out> ByRef pmdVal As MpegDuration) As Integer

		<PreserveSig> _
		Function GetRecordTitleText(<[In]> dwRecordIndex As Integer, <Out> ByRef pdwLength As Integer, <Out> ByRef ppText As IntPtr) As Integer

		<PreserveSig> _
		Function GetRecordCountOfDescriptors(<[In]> dwRecordIndex As Integer, <Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByIndex(<[In]> dwRecordIndex As Integer, <[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetRecordDescriptorByTag(<[In]> dwRecordIndex As Integer, <[In]> bTag As Byte, <[In], Out> pdwCookie As DsInt, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("5a142cc9-b8cf-4a86-a040-e9cadf3ef3e7"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IATSC_ETT
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList, <[In]> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetProtocolVersion(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetEtmId(<Out> ByRef pdwVal As Integer) As Integer

		Function GetExtendedMessageText(<Out> ByRef pdwLength As Integer, <Out> ByRef ppText As IntPtr) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("6bf42423-217d-4d6f-81e1-3a7b360ec896"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IATSC_STT
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList, <[In]> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetProtocolVersion(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSystemTime(<Out> ByRef pmdtSystemTime As MpegDateAndTime) As Integer

		<PreserveSig> _
		Function GetGpsUtcOffset(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetDaylightSavings(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetCountOfTableDescriptors(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByIndex(<[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByTag(<[In]> bTag As Byte, <[In]> pdwCookie As IntPtr, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("1FF544D6-161D-4fae-9FAA-4F9F492AE999"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface ISCTE_EAS
		<PreserveSig> _
		Function Initialize(<[In]> pSectionList As ISectionList, <[In]> pMPEGData As IMpeg2Data) As Integer

		<PreserveSig> _
		Function GetVersionNumber(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetSequencyNumber(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetProtocolVersion(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetEASEventID(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetOriginatorCode(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetEASEventCodeLen(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetEASEventCode(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRawNatureOfActivationTextLen(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRawNatureOfActivationText(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetNatureOfActivationText(<[In], MarshalAs(UnmanagedType.BStr)> bstrIS0639code As String, <Out, MarshalAs(UnmanagedType.BStr)> ByRef pbstrString As String) As Integer

		<PreserveSig> _
		Function GetTimeRemaining(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetStartTime(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetDuration(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetAlertPriority(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetDetailsOOBSourceID(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetDetailsMajor(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetDetailsMinor(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetDetailsAudioOOBSourceID(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetAlertText(<[In], MarshalAs(UnmanagedType.BStr)> bstrIS0639code As String, <Out, MarshalAs(UnmanagedType.BStr)> ByRef pbstrString As String) As Integer

		<PreserveSig> _
		Function GetRawAlertTextLen(<Out> ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetRawAlertText(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLocationCount(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLocationCodes(<[In]> bIndex As Byte, <Out> ByRef pbState As Byte, <Out> ByRef pbCountySubdivision As Byte, <Out> ByRef pwCounty As Short) As Integer

		<PreserveSig> _
		Function GetExceptionCount(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetExceptionService(<[In]> bIndex As Byte, <Out> ByRef pbIBRef As Byte, <Out> ByRef pwFirst As Byte, <Out> ByRef pwSecond As Short) As Integer

		<PreserveSig> _
		Function GetCountOfTableDescriptors(<Out> ByRef pdwVal As Integer) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByIndex(<[In]> dwIndex As Integer, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer

		<PreserveSig> _
		Function GetTableDescriptorByTag(<[In]> bTag As Byte, <[In]> pdwCookie As IntPtr, <Out> ByRef ppDescriptor As IGenericDescriptor) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("FF76E60C-0283-43ea-BA32-B422238547EE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IAtscContentAdvisoryDescriptor
		<PreserveSig> _
		Function GetTag(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetLength(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRatingRegionCount(<Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordRatingRegion(<[In]> bIndex As Byte, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordRatedDimensions(<[In]> bIndex As Byte, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordRatingDimension(<[In]> bIndexOuter As Byte, <[In]> bIndexInner As Byte, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordRatingValue(<[In]> bIndexOuter As Byte, <[In]> bIndexInner As Byte, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetRecordRatingDescriptionText(<[In]> bIndex As Byte, <Out> ByRef pbLength As Byte, <Out> ByRef ppText As IntPtr) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("40834007-6834-46f0-BD45-D5F6A6BE258C"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface ICaptionServiceDescriptor
		<PreserveSig> _
		Function GetNumberOfServices(<Out> ByRef pbVal As Byte) As Integer

			' probably a byte[3]
		<PreserveSig> _
		Function GetLanguageCode(<[In]> bIndex As Byte, <Out> ByRef LangCode As Integer) As Integer

		<PreserveSig> _
		Function GetCaptionServiceNumber(<[In]> bIndex As Byte, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetCCType(<[In]> bIndex As Byte, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetEasyReader(<[In]> bIndex As Byte, <Out> ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetWideAspectRatio(<[In]> bIndex As Byte, <Out> ByRef pbVal As Byte) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("58C3C827-9D91-4215-BFF3-820A49F0904C"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IServiceLocationDescriptor
		<PreserveSig> _
		Function GetPCR_PID(ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetNumberOfElements(ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetElementStreamType(bIndex As Byte, ByRef pbVal As Byte) As Integer

		<PreserveSig> _
		Function GetElementPID(bIndex As Byte, ByRef pwVal As Short) As Integer

		<PreserveSig> _
		Function GetElementLanguageCode(bIndex As Byte, <MarshalAs(UnmanagedType.LPArray, SizeConst := 3)> ByRef LangCode As Byte()) As Integer

	End Interface

#End If

End Namespace
