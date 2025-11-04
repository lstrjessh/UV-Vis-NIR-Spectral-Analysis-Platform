
Imports System.Runtime.InteropServices

Namespace BDA

#If ALLOW_UNTESTED_INTERFACES Then

	Public Enum BDAEventID
		SignalLoss = 0
		SignalLock
		DataStart
		DataStop
		ChannelAcquired
		ChannelLost
		ChannelSourceChanged
		ChannelActivated
		ChannelDeactivated
		SubChannelAcquired
		SubChannelLost
		SubChannelSourceChanged
		SubChannelActivated
		SubChannelDeactivated
		AccessGranted
		AccessDenied
		OfferExtended
		PurchaseCompleted
		SmartCardInserted
		SmartCardRemoved
	End Enum

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure BDATemplatePinJoint
		Public uliTemplateConnection As Integer
		Public ulcInstancesMax As Integer
	End Structure

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure KSBDAFrameInfo
		Public ExtendedHeaderSize As Integer ' Size of this extended header
		Public dwFrameFlags As Integer
		Public ulEvent As Integer
		Public ulChannelNumber As Integer
		Public ulSubchannelNumber As Integer
		Public ulReason As Integer
	End Structure

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure MPEG2TransportStride
		Public dwOffset As Integer
		Public dwPacketLength As Integer
		Public dwStride As Integer
	End Structure

	<Flags> _
	Public Enum ScanModulationTypes
		ScanMod16QAM = &H1
		ScanMod32QAM = &H2
		ScanMod64QAM = &H4
		ScanMod80QAM = &H8
		ScanMod96QAM = &H10
		ScanMod112QAM = &H20
		ScanMod128QAM = &H40
		ScanMod160QAM = &H80
		ScanMod192QAM = &H100
		ScanMod224QAM = &H200
		ScanMod256QAM = &H400
		ScanMod320QAM = &H800
		ScanMod384QAM = &H1000
		ScanMod448QAM = &H2000
		ScanMod512QAM = &H4000
		ScanMod640QAM = &H8000
		ScanMod768QAM = &H10000
		ScanMod896QAM = &H20000
		ScanMod1024QAM = &H40000
		ScanModQPSK = &H80000
		ScanModBPSK = &H100000
		ScanModOQPSK = &H200000
		ScanMod8VSB = &H400000
		ScanMod16VSB = &H800000
		ScanModAM_RADIO = &H1000000
		ScanModFM_RADIO = &H2000000
		ScanMod8PSK = &H4000000
		ScanModRF = &H8000000
		MCEDigitalCable = ModulationType.Mod640Qam Or ModulationType.Mod256Qam
		MCETerrestrialATSC = ModulationType.Mod8Vsb
		MCEAnalogTv = ModulationType.ModRF
        MCEAll_TV = -1
	End Enum

	Public Enum RollOff
		NotSet = -1
		NotDefined = 0
		Twenty = 1
		TwentyFive
		ThirtyFive
		Max
	End Enum

	Public Enum Pilot
		NotSet = -1
		NotDefined = 0
		Off = 1
		[On]
		Max
	End Enum

	Public Enum ApplicationTypeType
		SCTE28ConditionalAccess = 0
		SCTE28PODHostBindingInformation
		SCTE28IPService
		SCTE28NetworkInterfaceSCTE55_2
		SCTE28NetworkInterfaceSCTE55_1
		SCTE28CopyProtection
		SCTE28Diagnostic
		SCTE28Undesignated
		SCTE28Reserved
	End Enum

#End If

    Public Enum FECMethod
        MethodNotSet = -1
        MethodNotDefined = 0
        Viterbi = 1     ' FEC is a Viterbi Binary Convolution.
        RS204_188       ' The FEC is Reed-Solomon 204/188 (outer FEC)
        Ldpc
        Bch
        RS147_130
        Max
    End Enum

    Public Enum BinaryConvolutionCodeRate
        RateNotSet = -1
        RateNotDefined = 0
        Rate1_2 = 1         ' 1/2
        Rate2_3             ' 2/3
        Rate3_4             ' 3/4
        Rate3_5
        Rate4_5
        Rate5_6             ' 5/6
        Rate5_11
        Rate7_8             ' 7/8
        Rate1_4
        Rate1_3
        Rate2_5
        Rate6_7
        Rate8_9
        Rate9_10
        RateMax
    End Enum

    Public Enum Polarisation
        NotSet = -1
        NotDefined = 0
        LinearH = 1 ' Linear horizontal polarisation
        LinearV     ' Linear vertical polarisation
        CircularL   ' Circular left polarisation
        CircularR   ' Circular right polarisation
        Max
    End Enum

    Public Enum SpectralInversion
        NotSet = -1
        NotDefined = 0
        Automatic = 1
        Normal
        Inverted
        Max
    End Enum

    Public Enum ModulationType
        ModNotSet = -1
        ModNotDefined = 0
        Mod16Qam = 1
        Mod32Qam
        Mod64Qam
        Mod80Qam
        Mod96Qam
        Mod112Qam
        Mod128Qam
        Mod160Qam
        Mod192Qam
        Mod224Qam
        Mod256Qam
        Mod320Qam
        Mod384Qam
        Mod448Qam
        Mod512Qam
        Mod640Qam
        Mod768Qam
        Mod896Qam
        Mod1024Qam
        ModQpsk
        ModBpsk
        ModOqpsk
        Mod8Vsb
        Mod16Vsb
        ModAnalogAmplitude  ' std am
        ModAnalogFrequency  ' std fm
        Mod8Psk
        ModRF
        Mod16Apsk
        Mod32Apsk
        ModNbcQpsk
        ModNbc8Psk
        ModDirectTv
        ModMax
    End Enum

    Public Enum DVBSystemType
        Cable
        Terrestrial
        Satellite
    End Enum

    Public Enum HierarchyAlpha
        HAlphaNotSet = -1
        HAlphaNotDefined = 0
        HAlpha1 = 1     ' Hierarchy alpha is 1.
        HAlpha2         ' Hierarchy alpha is 2.
        HAlpha4         ' Hierarchy alpha is 4.
        HAlphaMax
    End Enum

    Public Enum GuardInterval
        GuardNotSet = -1
        GuardNotDefined = 0
        Guard1_32 = 1   ' Guard interval is 1/32
        Guard1_16       ' Guard interval is 1/16
        Guard1_8        ' Guard interval is 1/8
        Guard1_4        ' Guard interval is 1/4
        GuardMax
    End Enum

    Public Enum TransmissionMode
        ModeNotSet = -1
        ModeNotDefined = 0
        Mode2K = 1      ' Transmission uses 1705 carriers (use a 2K FFT)
        Mode8K          ' Transmission uses 6817 carriers (use an 8K FFT)
        Mode4K
        Mode2KInterleaved
        Mode4KInterleaved
        ModeMax
    End Enum

    Public Enum ComponentStatus
        Active
        Inactive
        Unavailable
    End Enum

    Public Enum ComponentCategory
        NotSet = -1
        Other = 0
        Video
        Audio
        Text
        Data
    End Enum

    Public Enum MPEG2StreamType
        BdaUninitializedMpeg2StreamType = -1
        Reserved1 = &H0
        IsoIec11172_2_Video = &H1
        IsoIec13818_2_Video = &H2
        IsoIec11172_3_Audio = &H3
        IsoIec13818_3_Audio = &H4
        IsoIec13818_1_PrivateSection = &H5
        IsoIec13818_1_Pes = &H6
        IsoIec13522_Mheg = &H7
        AnnexADsmCC = &H8
        ItuTRecH222_1 = &H9
        IsoIec13818_6_TypeA = &HA
        IsoIec13818_6_TypeB = &HB
        IsoIec13818_6_TypeC = &HC
        IsoIec13818_6_TypeD = &HD
        IsoIec13818_1_Auxiliary = &HE
        IsoIec13818_1_Reserved = &HF
        UserPrivate = &H10
        IsoIecUserPrivate = &H80
        DolbyAc3Audio = &H81
    End Enum

    <Flags()> _
    Public Enum ATSCComponentTypeFlags
        None = &H0
        ATSCCT_AC3 = &H1
    End Enum

    <StructLayout(LayoutKind.Sequential)> _
    Public Structure BDATemplateConnection
        Public FromNodeType As Integer
        Public FromNodePinType As Integer
        Public ToNodeType As Integer
        Public ToNodePinType As Integer
    End Structure

    <Flags()> _
    Public Enum BDACompFlags
        NotDefined = &H0            ' BDACOMP_NOT_DEFINED
        ExcludeTSFromTR = &H1       ' BDACOMP_EXCLUDE_TS_FROM_TR
        IncludeLocatorInTR = &H2    ' BDACOMP_INCLUDE_LOCATOR_IN_TR
    End Enum

End Namespace
