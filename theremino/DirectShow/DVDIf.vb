
Imports System.Drawing
Imports System.Runtime.InteropServices

#If ALLOW_UNTESTED_INTERFACES Then

	Public Enum DVD_NavCmdType
		Pre = 1
		Post = 2
		Cell = 3
		Button = 4
	End Enum

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure DvdAtr
		Public ulCAT As Integer
		<MarshalAs(UnmanagedType.ByValArray, ArraySubType := UnmanagedType.I1, SizeConst := 768)> _
		Public registers As Byte()
	End Structure

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure DvdVideoATR
		<MarshalAs(UnmanagedType.ByValArray, ArraySubType := UnmanagedType.I1, SizeConst := 2)> _
		Public attributes As Byte()
	End Structure

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure DvdAudioATR
		<MarshalAs(UnmanagedType.ByValArray, ArraySubType := UnmanagedType.I1, SizeConst := 8)> _
		Public attributes As Byte()
	End Structure

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure DvdSubpictureATR
		<MarshalAs(UnmanagedType.ByValArray, ArraySubType := UnmanagedType.I1, SizeConst := 6)> _
		Public attributes As Byte()
	End Structure

	<StructLayout(LayoutKind.Sequential)> _
	Public Structure DvdPlaybackLocation
		Public TitleNum As Integer
		Public ChapterNum As Integer
		Public TimeCode As Integer
	End Structure

#End If

Public Enum DvdDomain
    FirstPlay = 1
    VideoManagerMenu
    VideoTitleSetMenu
    Title
    [Stop]
End Enum

Public Enum DvdMenuId
    Title = 2
    Root = 3
    Subpicture = 4
    Audio = 5
    Angle = 6
    Chapter = 7
End Enum

Public Enum DvdDiscSide
    SideA = 1
    SideB = 2
End Enum

Public Enum DvdPreferredDisplayMode
    DisplayContentDefault = 0
    Display16x9 = 1
    Display4x3PanScanPreferred = 2
    Display4x3LetterBoxPreferred = 3
End Enum

Public Enum DvdFrameRate
    FPS25 = 1
    FPS30NonDrop = 3
End Enum

<Flags()> _
Public Enum DvdTimeCodeFlags
    None = 0
    FPS25 = &H1
    FPS30 = &H2
    DropFrame = &H4
    Interpolated = &H8
End Enum

<Flags()> _
Public Enum ValidUOPFlag
    None = 0
    PlayTitleOrAtTime = &H1
    PlayChapter = &H2
    PlayTitle = &H4
    [Stop] = &H8
    ReturnFromSubMenu = &H10
    PlayChapterOrAtTime = &H20
    PlayPrevOrReplay_Chapter = &H40
    PlayNextChapter = &H80
    PlayForwards = &H100
    PlayBackwards = &H200
    ShowMenuTitle = &H400
    ShowMenuRoot = &H800
    ShowMenuSubPic = &H1000
    ShowMenuAudio = &H2000
    ShowMenuAngle = &H4000
    ShowMenuChapter = &H8000
    [Resume] = &H10000
    SelectOrActivateButton = &H20000
    StillOff = &H40000
    PauseOn = &H80000
    SelectAudioStream = &H100000
    SelectSubPicStream = &H200000
    SelectAngle = &H400000
    SelectKaraokeAudioPresentationMode = &H800000
    SelectVideoModePreference = &H1000000
End Enum

<Flags()> _
Public Enum DvdCmdFlags
    None = &H0
    Flush = &H1
    SendEvents = &H2
    Block = &H4
    StartWhenRendered = &H8
    EndAfterRendered = &H10
End Enum

Public Enum DvdOptionFlag
    ResetOnStop = 1
    NotifyParentalLevelChange = 2
    HMSFTimeCodeEvents = 3
    AudioDuringFFwdRew = 4
    EnableNonblockingAPIs = 5
    CacheSizeInMB = 6
    EnablePortableBookmarks = 7
    EnableExtendedCopyProtectErrors = 8
    NotifyPositionChange = 9
    IncreaseOutputControl = 10
    EnableStreaming = 11
    EnableESOutput = 12
    EnableTitleLength = 13
    DisableStillThrottle = 14
    EnableLoggingEvents = 15
    MaxReadBurstInKB = 16
    ReadBurstPeriodInMS = 17
End Enum

Public Enum DvdRelativeButton
    Upper = 1
    Lower = 2
    Left = 3
    Right = 4
End Enum

<Flags()> _
Public Enum DvdParentalLevel
    None = 0
    Level8 = &H8000
    Level7 = &H4000
    Level6 = &H2000
    Level5 = &H1000
    Level4 = &H800
    Level3 = &H400
    Level2 = &H200
    Level1 = &H100
End Enum

Public Enum DvdAudioLangExt
    NotSpecified = 0
    Captions = 1
    VisuallyImpaired = 2
    DirectorComments1 = 3
    DirectorComments2 = 4
End Enum

Public Enum DvdSubPictureLangExt
    NotSpecified = 0
    CaptionNormal = 1
    CaptionBig = 2
    CaptionChildren = 3
    CCNormal = 5
    CCBig = 6
    CCChildren = 7
    Forced = 9
    DirectorCommentsNormal = 13
    DirectorCommentsBig = 14
    DirectorCommentsChildren = 15
End Enum

Public Enum DvdAudioAppMode
    None = 0
    Karaoke = 1
    Surround = 2
    Other = 3
End Enum

Public Enum DvdAudioFormat
    AC3 = 0
    MPEG1 = 1
    MPEG1_DRC = 2
    MPEG2 = 3
    MPEG2_DRC = 4
    LPCM = 5
    DTS = 6
    SDDS = 7
    Other = 8
End Enum

<Flags()> _
Public Enum DvdKaraokeDownMix
    None = 0
    Mix_0to0 = &H1
    Mix_1to0 = &H2
    Mix_2to0 = &H4
    Mix_3to0 = &H8
    Mix_4to0 = &H10
    Mix_Lto0 = &H20
    Mix_Rto0 = &H40
    Mix_0to1 = &H100
    Mix_1to1 = &H200
    Mix_2to1 = &H400
    Mix_3to1 = &H800
    Mix_4to1 = &H1000
    Mix_Lto1 = &H2000
    Mix_Rto1 = &H4000
End Enum

<Flags()> _
Public Enum DvdKaraokeContents As Short
    None = 0
    GuideVocal1 = &H1
    GuideVocal2 = &H2
    GuideMelody1 = &H4
    GuideMelody2 = &H8
    GuideMelodyA = &H10
    GuideMelodyB = &H20
    SoundEffectA = &H40
    SoundEffectB = &H80
End Enum

Public Enum DvdKaraokeAssignment
    reserved0 = 0
    reserved1 = 1
    LR = 2
    LRM = 3
    LR1 = 4
    LRM1 = 5
    LR12 = 6
    LRM12 = 7
End Enum

Public Enum DvdVideoCompression
    Other = 0
    Mpeg1 = 1
    Mpeg2 = 2
End Enum

Public Enum DvdSubPictureType
    NotSpecified = 0
    Language = 1
    Other = 2
End Enum

Public Enum DvdSubPictureCoding
    RunLength = 0
    Extended = 1
    Other = 2
End Enum

Public Enum DvdTitleAppMode
    NotSpecified = 0
    Karaoke = 1
    Other = 3
End Enum

Public Enum DvdTextStringType
    DVD_Struct_Volume = &H1
    DVD_Struct_Title = &H2
    DVD_Struct_ParentalID = &H3
    DVD_Struct_PartOfTitle = &H4
    DVD_Struct_Cell = &H5
    DVD_Stream_Audio = &H10
    DVD_Stream_Subpicture = &H11
    DVD_Stream_Angle = &H12
    DVD_Channel_Audio = &H20
    DVD_General_Name = &H30
    DVD_General_Comments = &H31
    DVD_Title_Series = &H38
    DVD_Title_Movie = &H39
    DVD_Title_Video = &H3A
    DVD_Title_Album = &H3B
    DVD_Title_Song = &H3C
    DVD_Title_Other = &H3F
    DVD_Title_Sub_Series = &H40
    DVD_Title_Sub_Movie = &H41
    DVD_Title_Sub_Video = &H42
    DVD_Title_Sub_Album = &H43
    DVD_Title_Sub_Song = &H44
    DVD_Title_Sub_Other = &H47
    DVD_Title_Orig_Series = &H48
    DVD_Title_Orig_Movie = &H49
    DVD_Title_Orig_Video = &H4A
    DVD_Title_Orig_Album = &H4B
    DVD_Title_Orig_Song = &H4C
    DVD_Title_Orig_Other = &H4F
    DVD_Other_Scene = &H50
    DVD_Other_Cut = &H51
    DVD_Other_Take = &H52
End Enum

Public Enum DvdTextCharSet
    CharSet_Unicode = 0
    CharSet_ISO646 = 1
    CharSet_JIS_Roman_Kanji = 2
    CharSet_ISO8859_1 = 3
    CharSet_ShiftJIS_Kanji_Roman_Katakana = 4
End Enum

<Flags()> _
Public Enum DvdAudioCaps
    None = 0
    AC3 = &H1
    MPEG2 = &H2
    LPCM = &H4
    DTS = &H8
    SDDS = &H10
End Enum

<Flags()> _
Public Enum AMDvdGraphFlags
    None = 0
    HWDecPrefer = &H1
    HWDecOnly = &H2
    SWDecPrefer = &H4
    SWDecOnly = &H8
    NoVPE = &H100
    DoNotClear = &H200
    VMR9Only = &H800
    EVROnly = &H1000    ' only use EVR (otherwise fail) for rendering
    EVRQOS = &H2000     ' Enabled EVR Dynamic QoS
    AdaptGraph = &H4000 ' Adapt graph building to machine capbilities
    Mask = &HFFFF       ' only lower WORD is used/allowed
End Enum

<Flags()> _
Public Enum AMDvdStreamFlags
    None = &H0
    Video = &H1
    Audio = &H2
    SubPic = &H4
End Enum

<Flags()> _
Public Enum AMOverlayNotifyFlags
    None = 0
    VisibleChange = &H1
    SourceChange = &H2
    DestChange = &H4
End Enum

<StructLayout(LayoutKind.Sequential)> _
Public Structure GPRMArray
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.I2, SizeConst:=16)> _
    Public registers As Short()
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure SPRMArray
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.I2, SizeConst:=24)> _
    Public registers As Short()
End Structure

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Class DvdHMSFTimeCode
    Public bHours As Byte
    Public bMinutes As Byte
    Public bSeconds As Byte
    Public bFrames As Byte
End Class

<StructLayout(LayoutKind.Sequential, Pack:=1)> _
Public Structure DvdPlaybackLocation2
    Public TitleNum As Integer
    Public ChapterNum As Integer
    Public TimeCode As DvdHMSFTimeCode
    Public TimeCodeFlags As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure DvdAudioAttributes
    Public AppMode As DvdAudioAppMode
    Public AppModeData As Byte
    Public AudioFormat As DvdAudioFormat
    Public Language As Integer
    Public LanguageExtension As DvdAudioLangExt
    <MarshalAs(UnmanagedType.Bool)> _
    Public fHasMultichannelInfo As Boolean
    Public dwFrequency As Integer
    Public bQuantization As Byte
    Public bNumberOfChannels As Byte
    Public dwReserved1 As Integer
    Public dwReserved2 As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure DvdMUAMixingInfo
    <MarshalAs(UnmanagedType.Bool)> _
    Public fMixTo0 As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public fMixTo1 As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public fMix0InPhase As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public fMix1InPhase As Boolean
    Public dwSpeakerPosition As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure DvdMUACoeff
    Public log2_alpha As Double
    Public log2_beta As Double
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure DvdMultichannelAudioAttributes
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.Struct, SizeConst:=8)> _
    Public Info As DvdMUAMixingInfo()
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.Struct, SizeConst:=8)> _
    Public Coeff As DvdMUACoeff()
End Structure

<StructLayout(LayoutKind.Sequential, Pack:=1, Size:=32)> _
Public Class DvdKaraokeAttributes
    Public bVersion As Byte
    Public fMasterOfCeremoniesInGuideVocal1 As Boolean
    Public fDuet As Boolean
    Public ChannelAssignment As DvdKaraokeAssignment
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.I2, SizeConst:=8)> _
    Public wChannelContents As DvdKaraokeContents()
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Structure DvdVideoAttributes
    <MarshalAs(UnmanagedType.Bool)> _
    Public panscanPermitted As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public letterboxPermitted As Boolean
    Public aspectX As Integer
    Public aspectY As Integer
    Public frameRate As Integer
    Public frameHeight As Integer
    Public compression As DvdVideoCompression
    <MarshalAs(UnmanagedType.Bool)> _
    Public line21Field1InGOP As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public line21Field2InGOP As Boolean
    Public sourceResolutionX As Integer
    Public sourceResolutionY As Integer
    <MarshalAs(UnmanagedType.Bool)> _
    Public isSourceLetterboxed As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public isFilmMode As Boolean
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure DvdSubpictureAttributes
    Public Type As DvdSubPictureType
    Public CodingMode As DvdSubPictureCoding
    Public Language As Integer
    Public LanguageExtension As DvdSubPictureLangExt
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Class DvdTitleAttributes
    Public AppMode As DvdTitleAppMode
    Public VideoAttributes As DvdVideoAttributes
    Public ulNumberOfAudioStreams As Integer
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.Struct, SizeConst:=8)> _
    Public AudioAttributes As DvdAudioAttributes()
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.Struct, SizeConst:=8)> _
    Public MultichannelAudioAttributes As DvdMultichannelAudioAttributes()
    Public ulNumberOfSubpictureStreams As Integer
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.Struct, SizeConst:=32)> _
    Public SubpictureAttributes As DvdSubpictureAttributes()
End Class

<StructLayout(LayoutKind.Sequential)> _
Public Structure DvdMenuAttributes
    <MarshalAs(UnmanagedType.ByValArray, ArraySubType:=UnmanagedType.Bool, SizeConst:=8)> _
    Public fCompatibleRegion As Boolean()
    Public VideoAttributes As DvdVideoAttributes
    <MarshalAs(UnmanagedType.Bool)> _
    Public fAudioPresent As Boolean
    Public AudioAttributes As DvdAudioAttributes
    <MarshalAs(UnmanagedType.Bool)> _
    Public fSubpicturePresent As Boolean
    Public SubpictureAttributes As DvdSubpictureAttributes
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure DvdDecoderCaps
    Public dwSize As Integer
    Public dwAudioCaps As DvdAudioCaps
    Public dFwdMaxRateVideo As Double
    Public dFwdMaxRateAudio As Double
    Public dFwdMaxRateSP As Double
    Public dBwdMaxRateVideo As Double
    Public dBwdMaxRateAudio As Double
    Public dBwdMaxRateSP As Double
    Public dwRes1 As Integer
    Public dwRes2 As Integer
    Public dwRes3 As Integer
    Public dwRes4 As Integer
End Structure

<StructLayout(LayoutKind.Sequential)> _
Public Structure AMDvdRenderStatus
    Public hrVPEStatus As Integer
    <MarshalAs(UnmanagedType.Bool)> _
    Public bDvdVolInvalid As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public bDvdVolUnknown As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public bNoLine21In As Boolean
    <MarshalAs(UnmanagedType.Bool)> _
    Public bNoLine21Out As Boolean
    Public iNumStreams As Integer
    Public iNumStreamsFailed As Integer
    Public dwFailedStreamsFlag As AMDvdStreamFlags
End Structure


#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("A70EFE61-E2A3-11d0-A9BE-00AA0061BE93"), Obsolete("The IDvdControl interface is deprecated. Use IDvdControl2 instead.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvdControl
		<PreserveSig> _
		Function TitlePlay(<[In]> ulTitle As Integer) As Integer

		<PreserveSig> _
		Function ChapterPlay(<[In]> ulTitle As Integer, <[In]> ulChapter As Integer) As Integer

		<PreserveSig> _
		Function TimePlay(<[In]> ulTitle As Integer, <[In]> bcdTime As Integer) As Integer

		<PreserveSig> _
		Function StopForResume() As Integer

		<PreserveSig> _
		Function GoUp() As Integer

		<PreserveSig> _
		Function TimeSearch(<[In]> bcdTime As Integer) As Integer

		<PreserveSig> _
		Function ChapterSearch(<[In]> ulChapter As Integer) As Integer

		<PreserveSig> _
		Function PrevPGSearch() As Integer

		<PreserveSig> _
		Function TopPGSearch() As Integer

		<PreserveSig> _
		Function NextPGSearch() As Integer

		<PreserveSig> _
		Function ForwardScan(<[In]> dwSpeed As Double) As Integer

		<PreserveSig> _
		Function BackwardScan(<[In]> dwSpeed As Double) As Integer

		<PreserveSig> _
		Function MenuCall(<[In]> MenuID As DvdMenuId) As Integer

		<PreserveSig> _
		Function [Resume]() As Integer

		<PreserveSig> _
		Function UpperButtonSelect() As Integer

		<PreserveSig> _
		Function LowerButtonSelect() As Integer

		<PreserveSig> _
		Function LeftButtonSelect() As Integer

		<PreserveSig> _
		Function RightButtonSelect() As Integer

		<PreserveSig> _
		Function ButtonActivate() As Integer

		<PreserveSig> _
		Function ButtonSelectAndActivate(<[In]> ulButton As Integer) As Integer

		<PreserveSig> _
		Function StillOff() As Integer

		<PreserveSig> _
		Function PauseOn() As Integer

		<PreserveSig> _
		Function PauseOff() As Integer

		<PreserveSig> _
		Function MenuLanguageSelect(<[In]> Language As Integer) As Integer

		<PreserveSig> _
		Function AudioStreamChange(<[In]> ulAudio As Integer) As Integer

		<PreserveSig> _
		Function SubpictureStreamChange(<[In]> ulSubPicture As Integer, <[In], MarshalAs(UnmanagedType.Bool)> bDisplay As Boolean) As Integer

		<PreserveSig> _
		Function AngleChange(<[In]> ulAngle As Integer) As Integer

		<PreserveSig> _
		Function ParentalLevelSelect(<[In]> ulParentalLevel As Integer) As Integer

		<PreserveSig> _
		Function ParentalCountrySelect(<[In]> wCountry As Short) As Integer

		<PreserveSig> _
		Function KaraokeAudioPresentationModeChange(<[In]> ulMode As Integer) As Integer

		<PreserveSig> _
		Function VideoModePreferrence(<[In]> ulPreferredDisplayMode As Integer) As Integer

		<PreserveSig> _
		Function SetRoot(<[In], MarshalAs(UnmanagedType.LPWStr)> pszPath As String) As Integer

		<PreserveSig> _
		Function MouseActivate(<[In]> point As Point) As Integer

		<PreserveSig> _
		Function MouseSelect(<[In]> point As Point) As Integer

		<PreserveSig> _
		Function ChapterPlayAutoStop(<[In]> ulTitle As Integer, <[In]> ulChapter As Integer, <[In]> ulChaptersToPlay As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("A70EFE60-E2A3-11d0-A9BE-00AA0061BE93"), Obsolete("The IDvdInfo interface is deprecated. Use IDvdInfo2 instead.", False), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDvdInfo
		<PreserveSig> _
		Function GetCurrentDomain(<Out> ByRef pDomain As DvdDomain) As Integer

		<PreserveSig> _
		Function GetCurrentLocation(<Out> ByRef pLocation As DvdPlaybackLocation) As Integer

		<PreserveSig> _
		Function GetTotalTitleTime(<Out> ByRef pulTotalTime As Integer) As Integer

		<PreserveSig> _
		Function GetCurrentButton(<Out> ByRef pulButtonsAvailable As Integer, <Out> ByRef pulCurrentButton As Integer) As Integer

		<PreserveSig> _
		Function GetCurrentAngle(<Out> ByRef pulAnglesAvailable As Integer, <Out> ByRef pulCurrentAngle As Integer) As Integer

		<PreserveSig> _
		Function GetCurrentAudio(<Out> ByRef pulStreamsAvailable As Integer, <Out> ByRef pulCurrentStream As Integer) As Integer

		<PreserveSig> _
		Function GetCurrentSubpicture(<Out> ByRef pulStreamsAvailable As Integer, <Out> ByRef pulCurrentStream As Integer, <Out, MarshalAs(UnmanagedType.Bool)> ByRef pIsDisabled As Boolean) As Integer

		<PreserveSig> _
		Function GetCurrentUOPS(<Out> ByRef pUOP As Integer) As Integer

		<PreserveSig> _
		Function GetAllSPRMs(<Out> ByRef pRegisterArray As SPRMArray) As Integer

		<PreserveSig> _
		Function GetAllGPRMs(<Out> ByRef pRegisterArray As GPRMArray) As Integer

		<PreserveSig> _
		Function GetAudioLanguage(<[In]> ulStream As Integer, <Out> ByRef pLanguage As Integer) As Integer

		<PreserveSig> _
		Function GetSubpictureLanguage(<[In]> ulStream As Integer, <Out> ByRef pLanguage As Integer) As Integer

		<PreserveSig> _
		Function GetTitleAttributes(<[In]> ulTitle As Integer, <Out> ByRef pATR As DvdAtr) As Integer

		<PreserveSig> _
		Function GetVMGAttributes(<Out> ByRef pATR As DvdAtr) As Integer

		<PreserveSig> _
		Function GetCurrentVideoAttributes(<Out> ByRef pATR As DvdVideoATR) As Integer

		<PreserveSig> _
		Function GetCurrentAudioAttributes(<Out> ByRef pATR As DvdAudioATR) As Integer

		<PreserveSig> _
		Function GetCurrentSubpictureAttributes(<Out> ByRef pATR As DvdSubpictureATR) As Integer


		<PreserveSig> _
		Function GetCurrentVolumeInfo(<Out> ByRef pulNumOfVol As Integer, <Out> ByRef pulThisVolNum As Integer, <Out> pSide As DvdDiscSide, <Out> ByRef pulNumOfTitles As Integer) As Integer


		' BYTE *
		<PreserveSig> _
		Function GetDVDTextInfo(<Out> ByRef pTextManager As IntPtr, <[In]> ulBufSize As Integer, <Out> ByRef pulActualSize As Integer) As Integer

		<PreserveSig> _
		Function GetPlayerParentalLevel(<Out> ByRef pulParentalLevel As Integer, <Out> ByRef pulCountryCode As Integer) As Integer

		<PreserveSig> _
		Function GetNumberOfChapters(<[In]> ulTitle As Integer, <Out> ByRef pulNumberOfChapters As Integer) As Integer

		<PreserveSig> _
		Function GetTitleParentalLevels(<[In]> ulTitle As Integer, <Out> ByRef pulParentalLevels As Integer) As Integer

		' LPSTR
		<PreserveSig> _
		Function GetRoot(<Out> ByRef pRoot As IntPtr, <[In]> ulBufSize As Integer, <Out> ByRef pulActualSize As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("153ACC21-D83B-11d1-82BF-00A0C9696C8F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDDrawExclModeVideo
		<PreserveSig> _
		Function SetDDrawObject(<[In], MarshalAs(UnmanagedType.IUnknown)> pDDrawObject As Object) As Integer

		<PreserveSig> _
		Function GetDDrawObject(<Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppDDrawObject As Object, <Out, MarshalAs(UnmanagedType.Bool)> ByRef pbUsingExternal As Boolean) As Integer

		<PreserveSig> _
		Function SetDDrawSurface(<[In], MarshalAs(UnmanagedType.IUnknown)> pDDrawSurface As Object) As Integer

		<PreserveSig> _
		Function GetDDrawSurface(<Out, MarshalAs(UnmanagedType.IUnknown)> ByRef ppDDrawSurface As Object, <Out, MarshalAs(UnmanagedType.Bool)> ByRef pbUsingExternal As Boolean) As Integer

		<PreserveSig> _
		Function SetDrawParameters(<[In]> prcSource As Rectangle, <[In]> prcTarget As Rectangle) As Integer

		<PreserveSig> _
		Function GetNativeVideoProps(<Out> ByRef pdwVideoWidth As Integer, <Out> ByRef pdwVideoHeight As Integer, <Out> ByRef pdwPictAspectRatioX As Integer, <Out> ByRef pdwPictAspectRatioY As Integer) As Integer

		<PreserveSig> _
		Function SetCallbackInterface(<[In], MarshalAs(UnmanagedType.IUnknown)> pCallback As Object, <[In]> dwFlags As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("913c24a0-20ab-11d2-9038-00a0c9697298"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDDrawExclModeVideoCallback
		<PreserveSig> _
		Function OnUpdateOverlay(<[In], MarshalAs(UnmanagedType.Bool)> bBefore As Boolean, <[In]> dwFlags As AMOverlayNotifyFlags, <[In], MarshalAs(UnmanagedType.Bool)> bOldVisible As Boolean, <[In]> prcOldSrc As Rectangle, <[In]> prcOldDest As Rectangle, <[In], MarshalAs(UnmanagedType.Bool)> bNewVisible As Boolean, _
			<[In]> prcNewSrc As Rectangle, <[In]> prcNewDest As Rectangle) As Integer

		<PreserveSig> _
		Function OnUpdateColorKey(<[In]> pKey As ColorKey, <[In]> dwColor As Integer) As Integer

		<PreserveSig> _
		Function OnUpdateSize(<[In]> dwWidth As Integer, <[In]> dwHeight As Integer, <[In]> dwARWidth As Integer, <[In]> dwARHeight As Integer) As Integer
	End Interface
#End If

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("FCC152B6-F372-11d0-8E00-00C04FD7C08B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDvdGraphBuilder
    <PreserveSig()> _
    Function GetFiltergraph(<Out()> ByRef ppGB As IGraphBuilder) As Integer

    <PreserveSig()> _
    Function GetDvdInterface(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal riid As Guid, <Out(), MarshalAs(UnmanagedType.[Interface])> ByRef ppvIF As Object) As Integer

    <PreserveSig()> _
    Function RenderDvdVideoVolume(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal lpcwszPathName As String, <[In]()> ByVal dwFlags As AMDvdGraphFlags, <Out()> ByRef pStatus As AMDvdRenderStatus) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("33BC7430-EEC0-11D2-8201-00A0C9D74842"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDvdControl2
    <PreserveSig()> _
    Function PlayTitle(<[In]()> ByVal ulTitle As Integer, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function PlayChapterInTitle(<[In]()> ByVal ulTitle As Integer, <[In]()> ByVal ulChapter As Integer, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function PlayAtTimeInTitle(<[In]()> ByVal ulTitle As Integer, <[In]()> ByVal pStartTime As DvdHMSFTimeCode, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function [Stop]() As Integer

    <PreserveSig()> _
    Function ReturnFromSubmenu(<[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function PlayAtTime(<[In]()> ByVal pTime As DvdHMSFTimeCode, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function PlayChapter(<[In]()> ByVal ulChapter As Integer, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function PlayPrevChapter(<[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function ReplayChapter(<[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function PlayNextChapter(<[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function PlayForwards(<[In]()> ByVal dSpeed As Double, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function PlayBackwards(<[In]()> ByVal dSpeed As Double, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function ShowMenu(<[In]()> ByVal MenuID As DvdMenuId, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function [Resume](<[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function SelectRelativeButton(ByVal buttonDir As DvdRelativeButton) As Integer

    <PreserveSig()> _
    Function ActivateButton() As Integer

    <PreserveSig()> _
    Function SelectButton(<[In]()> ByVal ulButton As Integer) As Integer

    <PreserveSig()> _
    Function SelectAndActivateButton(<[In]()> ByVal ulButton As Integer) As Integer

    <PreserveSig()> _
    Function StillOff() As Integer

    <PreserveSig()> _
    Function Pause(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bState As Boolean) As Integer

    <PreserveSig()> _
    Function SelectAudioStream(<[In]()> ByVal ulAudio As Integer, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function SelectSubpictureStream(<[In]()> ByVal ulSubPicture As Integer, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function SetSubpictureState(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bState As Boolean, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function SelectAngle(<[In]()> ByVal ulAngle As Integer, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function SelectParentalLevel(<[In]()> ByVal ulParentalLevel As Integer) As Integer

    <PreserveSig()> _
    Function SelectParentalCountry(<[In](), MarshalAs(UnmanagedType.LPArray)> ByVal bCountry As Byte()) As Integer

    <PreserveSig()> _
    Function SelectKaraokeAudioPresentationMode(<[In]()> ByVal ulMode As DvdKaraokeDownMix) As Integer

    <PreserveSig()> _
    Function SelectVideoModePreference(<[In]()> ByVal ulPreferredDisplayMode As DvdPreferredDisplayMode) As Integer

    <PreserveSig()> _
    Function SetDVDDirectory(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszwPath As String) As Integer

    <PreserveSig()> _
    Function ActivateAtPosition(<[In]()> ByVal point As Point) As Integer

    <PreserveSig()> _
    Function SelectAtPosition(<[In]()> ByVal point As Point) As Integer

    <PreserveSig()> _
    Function PlayChaptersAutoStop(<[In]()> ByVal ulTitle As Integer, <[In]()> ByVal ulChapter As Integer, <[In]()> ByVal ulChaptersToPlay As Integer, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function AcceptParentalLevelChange(<[In](), MarshalAs(UnmanagedType.Bool)> ByVal bAccept As Boolean) As Integer

    <PreserveSig()> _
    Function SetOption(<[In]()> ByVal flag As DvdOptionFlag, <[In](), MarshalAs(UnmanagedType.Bool)> ByVal fState As Boolean) As Integer

    <PreserveSig()> _
    Function SetState(<[In]()> ByVal pState As IDvdState, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function PlayPeriodInTitleAutoStop(<[In]()> ByVal ulTitle As Integer, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pStartTime As DvdHMSFTimeCode, <[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pEndTime As DvdHMSFTimeCode, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function SetGPRM(<[In]()> ByVal ulIndex As Integer, <[In]()> ByVal wValue As Short, <[In]()> ByVal dwFlags As DvdCmdFlags, <Out()> ByRef ppCmd As IDvdCmd) As Integer

    <PreserveSig()> _
    Function SelectDefaultMenuLanguage(<[In]()> ByVal Language As Integer) As Integer

    <PreserveSig()> _
    Function SelectDefaultAudioLanguage(<[In]()> ByVal Language As Integer, <[In]()> ByVal audioExtension As DvdAudioLangExt) As Integer

    <PreserveSig()> _
    Function SelectDefaultSubpictureLanguage(<[In]()> ByVal Language As Integer, <[In]()> ByVal subpictureExtension As DvdSubPictureLangExt) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("34151510-EEC0-11D2-8201-00A0C9D74842"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDvdInfo2
    <PreserveSig()> _
    Function GetCurrentDomain(<Out()> ByRef pDomain As DvdDomain) As Integer

    <PreserveSig()> _
    Function GetCurrentLocation(<Out()> ByRef pLocation As DvdPlaybackLocation2) As Integer

    <PreserveSig()> _
    Function GetTotalTitleTime(<Out()> ByVal pTotalTime As DvdHMSFTimeCode, <Out()> ByRef ulTimeCodeFlags As DvdTimeCodeFlags) As Integer

    <PreserveSig()> _
    Function GetCurrentButton(<Out()> ByRef pulButtonsAvailable As Integer, <Out()> ByRef pulCurrentButton As Integer) As Integer

    <PreserveSig()> _
    Function GetCurrentAngle(<Out()> ByRef pulAnglesAvailable As Integer, <Out()> ByRef pulCurrentAngle As Integer) As Integer

    <PreserveSig()> _
    Function GetCurrentAudio(<Out()> ByRef pulStreamsAvailable As Integer, <Out()> ByRef pulCurrentStream As Integer) As Integer

    <PreserveSig()> _
    Function GetCurrentSubpicture(<Out()> ByRef pulStreamsAvailable As Integer, <Out()> ByRef pulCurrentStream As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pbIsDisabled As Boolean) As Integer

    <PreserveSig()> _
    Function GetCurrentUOPS(<Out()> ByRef pulUOPs As ValidUOPFlag) As Integer

    <PreserveSig()> _
    Function GetAllSPRMs(<Out()> ByRef pRegisterArray As SPRMArray) As Integer

    <PreserveSig()> _
    Function GetAllGPRMs(<Out()> ByRef pRegisterArray As GPRMArray) As Integer

    <PreserveSig()> _
    Function GetAudioLanguage(<[In]()> ByVal ulStream As Integer, <Out()> ByRef pLanguage As Integer) As Integer

    <PreserveSig()> _
    Function GetSubpictureLanguage(<[In]()> ByVal ulStream As Integer, <Out()> ByRef pLanguage As Integer) As Integer

    <PreserveSig()> _
    Function GetTitleAttributes(<[In]()> ByVal ulTitle As Integer, <Out()> ByRef pMenu As DvdMenuAttributes, <[In](), Out(), MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef:=GetType(DTAMarshaler))> ByVal pTitle As DvdTitleAttributes) As Integer

    <PreserveSig()> _
    Function GetVMGAttributes(<Out()> ByRef pATR As DvdMenuAttributes) As Integer

    <PreserveSig()> _
    Function GetCurrentVideoAttributes(<Out()> ByRef pATR As DvdVideoAttributes) As Integer

    <PreserveSig()> _
    Function GetAudioAttributes(<[In]()> ByVal ulStream As Integer, <Out()> ByRef pATR As DvdAudioAttributes) As Integer

    <PreserveSig()> _
    Function GetKaraokeAttributes(<[In]()> ByVal ulStream As Integer, <[In](), Out(), MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef:=GetType(DKAMarshaler))> ByVal pAttributes As DvdKaraokeAttributes) As Integer

    <PreserveSig()> _
    Function GetSubpictureAttributes(<[In]()> ByVal ulStream As Integer, <Out()> ByRef pATR As DvdSubpictureAttributes) As Integer

    <PreserveSig()> _
    Function GetDVDVolumeInfo(<Out()> ByRef pulNumOfVolumes As Integer, <Out()> ByRef pulVolume As Integer, <Out()> ByRef pSide As DvdDiscSide, <Out()> ByRef pulNumOfTitles As Integer) As Integer

    <PreserveSig()> _
    Function GetDVDTextNumberOfLanguages(<Out()> ByRef pulNumOfLangs As Integer) As Integer

    <PreserveSig()> _
    Function GetDVDTextLanguageInfo(<[In]()> ByVal ulLangIndex As Integer, <Out()> ByRef pulNumOfStrings As Integer, <Out()> ByRef pLangCode As Integer, <Out()> ByRef pbCharacterSet As DvdTextCharSet) As Integer

    <PreserveSig()> _
    Function GetDVDTextStringAsNative(<[In]()> ByVal ulLangIndex As Integer, <[In]()> ByVal ulStringIndex As Integer, <MarshalAs(UnmanagedType.LPStr)> ByVal pbBuffer As System.Text.StringBuilder, <[In]()> ByVal ulMaxBufferSize As Integer, <Out()> ByRef pulActualSize As Integer, <Out()> ByRef pType As DvdTextStringType) As Integer

    <PreserveSig()> _
    Function GetDVDTextStringAsUnicode(<[In]()> ByVal ulLangIndex As Integer, <[In]()> ByVal ulStringIndex As Integer, ByVal pchwBuffer As System.Text.StringBuilder, <[In]()> ByVal ulMaxBufferSize As Integer, <Out()> ByRef pulActualSize As Integer, <Out()> ByRef pType As DvdTextStringType) As Integer

    <PreserveSig()> _
    Function GetPlayerParentalLevel(<Out()> ByRef pulParentalLevel As Integer, <Out(), MarshalAs(UnmanagedType.LPArray, SizeConst:=2)> ByVal pbCountryCode As Byte()) As Integer

    <PreserveSig()> _
    Function GetNumberOfChapters(<[In]()> ByVal ulTitle As Integer, <Out()> ByRef pulNumOfChapters As Integer) As Integer

    <PreserveSig()> _
    Function GetTitleParentalLevels(<[In]()> ByVal ulTitle As Integer, <Out()> ByRef pulParentalLevels As DvdParentalLevel) As Integer

    <PreserveSig()> _
    Function GetDVDDirectory(ByVal pszwPath As System.Text.StringBuilder, <[In]()> ByVal ulMaxSize As Integer, <Out()> ByRef pulActualSize As Integer) As Integer

    <PreserveSig()> _
    Function IsAudioStreamEnabled(<[In]()> ByVal ulStreamNum As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pbEnabled As Boolean) As Integer

    <PreserveSig()> _
    Function GetDiscID(<[In](), MarshalAs(UnmanagedType.LPWStr)> ByVal pszwPath As String, <Out()> ByRef pullDiscID As Long) As Integer

    <PreserveSig()> _
    Function GetState(<Out()> ByRef pStateData As IDvdState) As Integer

    <PreserveSig()> _
    Function GetMenuLanguages(<MarshalAs(UnmanagedType.LPArray)> ByVal pLanguages As Integer(), <[In]()> ByVal ulMaxLanguages As Integer, <Out()> ByRef pulActualLanguages As Integer) As Integer

    <PreserveSig()> _
    Function GetButtonAtPosition(<[In]()> ByVal point As Point, <Out()> ByRef pulButtonIndex As Integer) As Integer

    <PreserveSig()> _
    Function GetCmdFromEvent(<[In]()> ByVal lParam1 As IntPtr, <Out()> ByRef pCmdObj As IDvdCmd) As Integer

    <PreserveSig()> _
    Function GetDefaultMenuLanguage(<Out()> ByRef pLanguage As Integer) As Integer

    <PreserveSig()> _
    Function GetDefaultAudioLanguage(<Out()> ByRef pLanguage As Integer, <Out()> ByRef pAudioExtension As DvdAudioLangExt) As Integer

    <PreserveSig()> _
    Function GetDefaultSubpictureLanguage(<Out()> ByRef pLanguage As Integer, <Out()> ByRef pSubpictureExtension As DvdSubPictureLangExt) As Integer

    <PreserveSig()> _
    Function GetDecoderCaps(ByRef pCaps As DvdDecoderCaps) As Integer

    <PreserveSig()> _
    Function GetButtonRect(<[In]()> ByVal ulButton As Integer, <Out()> ByVal pRect As DsRect) As Integer

    <PreserveSig()> _
    Function IsSubpictureStreamEnabled(<[In]()> ByVal ulStreamNum As Integer, <Out(), MarshalAs(UnmanagedType.Bool)> ByRef pbEnabled As Boolean) As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("5a4a97e4-94ee-4a55-9751-74b5643aa27d"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDvdCmd
    <PreserveSig()> _
    Function WaitForStart() As Integer

    <PreserveSig()> _
    Function WaitForEnd() As Integer
End Interface

<ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("86303d6d-1c4a-4087-ab42-f711167048ef"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
Public Interface IDvdState
    <PreserveSig()> _
    Function GetDiscID(<Out()> ByRef pullUniqueID As Long) As Integer

    <PreserveSig()> _
    Function GetParentalLevel(<Out()> ByRef pulParentalLevel As Integer) As Integer
End Interface
