
Imports System.Runtime.InteropServices

Namespace BDA

#If ALLOW_UNTESTED_INTERFACES Then

	Public Enum EnTvRat_MPAA
		MPAA_NotApplicable = EnTvRat_GenericLevel.TvRat_0
		MPAA_G = EnTvRat_GenericLevel.TvRat_1
		MPAA_PG = EnTvRat_GenericLevel.TvRat_2
		MPAA_PG13 = EnTvRat_GenericLevel.TvRat_3
		MPAA_R = EnTvRat_GenericLevel.TvRat_4
		MPAA_NC17 = EnTvRat_GenericLevel.TvRat_5
		MPAA_X = EnTvRat_GenericLevel.TvRat_6
		MPAA_NotRated = EnTvRat_GenericLevel.TvRat_7
	End Enum

	Public Enum EnTvRat_US_TV
		US_TV_None = EnTvRat_GenericLevel.TvRat_0
		US_TV_Y = EnTvRat_GenericLevel.TvRat_1
		US_TV_Y7 = EnTvRat_GenericLevel.TvRat_2
		US_TV_G = EnTvRat_GenericLevel.TvRat_3
		US_TV_PG = EnTvRat_GenericLevel.TvRat_4
		US_TV_14 = EnTvRat_GenericLevel.TvRat_5
		US_TV_MA = EnTvRat_GenericLevel.TvRat_6
		US_TV_None7 = EnTvRat_GenericLevel.TvRat_7
	End Enum

	Public Enum EnTvRat_CAE_TV
		CAE_TV_Exempt = EnTvRat_GenericLevel.TvRat_0
		CAE_TV_C = EnTvRat_GenericLevel.TvRat_1
		CAE_TV_C8 = EnTvRat_GenericLevel.TvRat_2
		CAE_TV_G = EnTvRat_GenericLevel.TvRat_3
		CAE_TV_PG = EnTvRat_GenericLevel.TvRat_4
		CAE_TV_14 = EnTvRat_GenericLevel.TvRat_5
		CAE_TV_18 = EnTvRat_GenericLevel.TvRat_6
		CAE_TV_Reserved = EnTvRat_GenericLevel.TvRat_7
	End Enum

	Public Enum EnTvRat_CAF_TV
		CAF_TV_Exempt = EnTvRat_GenericLevel.TvRat_0
		CAF_TV_G = EnTvRat_GenericLevel.TvRat_1
		CAF_TV_8 = EnTvRat_GenericLevel.TvRat_2
		CAF_TV_13 = EnTvRat_GenericLevel.TvRat_3
		CAF_TV_16 = EnTvRat_GenericLevel.TvRat_4
		CAF_TV_18 = EnTvRat_GenericLevel.TvRat_5
		CAF_TV_Reserved6 = EnTvRat_GenericLevel.TvRat_6
		CAF_TV_Reserved = EnTvRat_GenericLevel.TvRat_7
	End Enum

	<Flags> _
	Public Enum BfEnTvRat_Attributes_US_TV
		None = 0
		IsBlocked = BfEnTvRat_GenericAttributes.BfIsBlocked
		IsViolent = BfEnTvRat_GenericAttributes.BfIsAttr_1
		IsSexualSituation = BfEnTvRat_GenericAttributes.BfIsAttr_2
		IsAdultLanguage = BfEnTvRat_GenericAttributes.BfIsAttr_3
		IsSexuallySuggestiveDialog = BfEnTvRat_GenericAttributes.BfIsAttr_4
		ValidAttrSubmask = 31
	End Enum

	<Flags> _
	Public Enum BfEnTvRat_Attributes_MPAA
		None = 0
		MPAA_IsBlocked = BfEnTvRat_GenericAttributes.BfIsBlocked
		MPAA_ValidAttrSubmask = 1
	End Enum

	<Flags> _
	Public Enum BfEnTvRat_Attributes_CAE_TV
		None = 0
		CAE_IsBlocked = BfEnTvRat_GenericAttributes.BfIsBlocked
		CAE_ValidAttrSubmask = 1
	End Enum

	<Flags> _
	Public Enum BfEnTvRat_Attributes_CAF_TV
		None = 0
		CAF_IsBlocked = BfEnTvRat_GenericAttributes.BfIsBlocked
		CAF_ValidAttrSubmask = 1
	End Enum

	<ComImport, Guid("C5C5C5F0-3ABC-11D6-B25B-00C04FA0C026")> _
	Public Class XDSToRat
	End Class

#End If


    Public Enum EnTvRat_System
        MPAA = 0
        US_TV = 1
        Canadian_English = 2
        Canadian_French = 3
        Reserved4 = 4
        System5 = 5
        System6 = 6
        Reserved7 = 7
        PBDA = 8
        AgeBased = 9
        TvRat_kSystems = 10
        TvRat_SystemDontKnow = 255
    End Enum

    Public Enum EnTvRat_GenericLevel
        TvRat_0 = 0
        TvRat_1 = 1
        TvRat_2 = 2
        TvRat_3 = 3
        TvRat_4 = 4
        TvRat_5 = 5
        TvRat_6 = 6
        TvRat_7 = 7
        TvRat_8 = 8
        TvRat_9 = 9
        TvRat_10 = 10
        TvRat_11 = 11
        TvRat_12 = 12
        TvRat_13 = 13
        TvRat_14 = 14
        TvRat_15 = 15
        TvRat_16 = 16
        TvRat_17 = 17
        TvRat_18 = 18
        TvRat_19 = 19
        TvRat_20 = 20
        TvRat_21 = 21
        TvRat_kLevels = 22
        TvRat_Unblock = -1
        TvRat_LevelDontKnow = 255
    End Enum

    <Flags()> _
    Public Enum BfEnTvRat_GenericAttributes
        BfAttrNone = 0
        BfIsBlocked = 1
        BfIsAttr_1 = 2
        BfIsAttr_2 = 4
        BfIsAttr_3 = 8
        BfIsAttr_4 = 16
        BfIsAttr_5 = 32
        BfIsAttr_6 = 64
        BfIsAttr_7 = 128
        BfValidAttrSubmask = 255
    End Enum

    <ComImport(), Guid("C5C5C5F1-3ABC-11D6-B25B-00C04FA0C026")> _
    Public Class EvalRat
    End Class



#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("C5C5C5B0-3ABC-11D6-B25B-00C04FA0C026"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
	Public Interface IXDSToRat
		<PreserveSig> _
		Function Init() As Integer

		<PreserveSig> _
		Function ParseXDSBytePair(<[In]> byte1 As Byte, <[In]> byte2 As Byte, <Out> ByRef pEnSystem As EnTvRat_System, <Out> ByRef pEnLevel As EnTvRat_GenericLevel, <Out> ByRef plBfEnAttributes As BfEnTvRat_GenericAttributes) As Integer
	End Interface

#End If

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("C5C5C5B1-3ABC-11D6-B25B-00C04FA0C026"), InterfaceType(ComInterfaceType.InterfaceIsDual)> _
    Public Interface IEvalRat
        <PreserveSig()> _
        Function get_BlockedRatingAttributes(<[In]()> ByVal enSystem As EnTvRat_System, <[In]()> ByVal enLevel As EnTvRat_GenericLevel, <Out()> ByRef plbfAttrs As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Function put_BlockedRatingAttributes(<[In]()> ByVal enSystem As EnTvRat_System, <[In]()> ByVal enLevel As EnTvRat_GenericLevel, <[In]()> ByVal plbfAttrs As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Function get_BlockUnRated(<Out(), MarshalAs(UnmanagedType.Bool)> ByRef pfBlockUnRatedShows As Boolean) As Integer

        <PreserveSig()> _
        Function put_BlockUnRated(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal pfBlockUnRatedShows As Boolean) As Integer

        <PreserveSig()> _
        Function MostRestrictiveRating(<[In]()> ByVal enSystem1 As EnTvRat_System, <[In]()> ByVal enEnLevel1 As EnTvRat_GenericLevel, <[In]()> ByVal lbfEnAttr1 As BfEnTvRat_GenericAttributes, <[In]()> ByVal enSystem2 As EnTvRat_System, <[In]()> ByVal enEnLevel2 As EnTvRat_GenericLevel, <[In]()> ByVal lbfEnAttr2 As BfEnTvRat_GenericAttributes, _
         <Out()> ByRef penSystem As EnTvRat_System, <Out()> ByRef penEnLevel As EnTvRat_GenericLevel, <Out()> ByRef plbfEnAttr As BfEnTvRat_GenericAttributes) As Integer

        <PreserveSig()> _
        Function TestRating(<[In]()> ByVal enShowSystem As EnTvRat_System, <[In]()> ByVal enShowLevel As EnTvRat_GenericLevel, <[In]()> ByVal lbfEnShowAttributes As BfEnTvRat_GenericAttributes) As Integer
    End Interface

End Namespace
