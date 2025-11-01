
Imports System.Runtime.InteropServices
Imports System.Text

Namespace DES

    Public NotInheritable Class DESResults
        ' Prevent people from trying to instantiate this class
        Private Sub New()
        End Sub

        Public Const E_NotInTree As Integer = &H80040400
        Public Const E_RenderEngineIsBroken As Integer = &H80040401
        Public Const E_MustInitRenderer As Integer = &H80040402
        Public Const E_NotDetermind As Integer = &H80040403
        Public Const E_NoTimeline As Integer = &H80040404
        Public Const S_WarnOutputReset As Integer = &H40404
    End Class


    Public NotInheritable Class DESError
        ' Prevent people from trying to instantiate this class
        Private Sub New()
        End Sub

        Public Shared Function GetErrorText(ByVal hr As Integer) As String
            Dim sRet As String = Nothing

            Select Case hr
                Case DESResults.E_NotInTree
                    sRet = "The object is not contained in the timeline."
                    Exit Select
                Case DESResults.E_RenderEngineIsBroken
                    sRet = "Operation failed because project was not rendered successfully."
                    Exit Select
                Case DESResults.E_MustInitRenderer
                    sRet = "Render engine has not been initialized."
                    Exit Select
                Case DESResults.E_NotDetermind
                    sRet = "Cannot determine requested value."
                    Exit Select
                Case DESResults.E_NoTimeline
                    sRet = "There is no timeline object."
                    Exit Select
                Case DESResults.S_WarnOutputReset
                    sRet = "The rendering portion of the graph was deleted. The application must rebuild it."
                    Exit Select
                Case Else
                    sRet = DsError.GetErrorText(hr)
                    Exit Select
            End Select

            Return sRet
        End Function

        ' If hr has a "failed" status code (E_*), throw an exception.  Note that status
        ' messages (S_*) are not considered failure codes.  If DES or DShow error text
        ' is available, it is used to build the exception, otherwise a generic com error
        ' is thrown.
        Public Shared Sub ThrowExceptionForHR(ByVal hr As Integer)
            ' If an error has occurred
            If hr < 0 Then
                ' If a string is returned, build a com error from it
                Dim buf As String = GetErrorText(hr)

                If buf IsNot Nothing Then
                    Throw New COMException(buf, hr)
                Else
                    ' No string, just use standard com error
                    Marshal.ThrowExceptionForHR(hr)
                End If
            End If
        End Sub
    End Class


    <ComImport(), Guid("78530B75-61F9-11D2-8CAD-00A024580902")> _
    Public Class AMTimeline
    End Class

    <ComImport(), Guid("ADF95821-DED7-11d2-ACBE-0080C75E246E")> _
    Public Class PropertySetter
    End Class

    <ComImport(), Guid("78530B78-61F9-11D2-8CAD-00A024580902")> _
    Public Class AMTimelineObj
    End Class

    <ComImport(), Guid("78530B7A-61F9-11D2-8CAD-00A024580902")> _
    Public Class AMTimelineSrc
    End Class

    <ComImport(), Guid("8F6C3C50-897B-11d2-8CFB-00A0C9441E20")> _
    Public Class AMTimelineTrack
    End Class

    <ComImport(), Guid("74D2EC80-6233-11d2-8CAD-00A024580902")> _
    Public Class AMTimelineComp
    End Class

    <ComImport(), Guid("F6D371E1-B8A6-11d2-8023-00C0DF10D434")> _
    Public Class AMTimelineGroup
    End Class

    <ComImport(), Guid("74D2EC81-6233-11d2-8CAD-00A024580902")> _
    Public Class AMTimelineTrans
    End Class

    <ComImport(), Guid("74D2EC82-6233-11d2-8CAD-00A024580902")> _
    Public Class AMTimelineEffect
    End Class

    <ComImport(), Guid("64D8A8E0-80A2-11d2-8CF3-00A0C9441E20")> _
    Public Class RenderEngine
    End Class

    <ComImport(), Guid("498B0949-BBE9-4072-98BE-6CCAEB79DC6F")> _
    Public Class SmartRenderEngine
    End Class

    <ComImport(), Guid("036A9790-C153-11d2-9EF7-006008039E37")> _
    Public Class AudMixer
    End Class

    <ComImport(), Guid("18C628EE-962A-11D2-8D08-00A0C9441E20")> _
    Public Class Xml2Dex
    End Class

    <ComImport(), Guid("CC1101F2-79DC-11D2-8CE6-00A0C9441E20")> _
    Public Class MediaLocator
    End Class

    <ComImport(), Guid("65BD0711-24D2-4ff7-9324-ED2E5D3ABAFA")> _
    Public Class MediaDet
    End Class

    <ComImport(), Guid("BB44391D-6ABD-422f-9E2E-385C9DFF51FC")> _
    Public Class DxtCompositor
    End Class

    <ComImport(), Guid("506D89AE-909A-44f7-9444-ABD575896E35")> _
    Public Class DxtAlphaSetter
    End Class

    <ComImport(), Guid("DE75D012-7A65-11D2-8CEA-00A0C9441E20")> _
    Public Class DxtJpeg
    End Class

    <ComImport(), Guid("0cfdd070-581a-11d2-9ee6-006008039e37")> _
    Public Class ColorSource
    End Class

    <ComImport(), Guid("C5B19592-145E-11d3-9F04-006008039E37")> _
    Public Class DxtKey
    End Class



#If ALLOW_UNTESTED_INTERFACES Then

	Public Enum DXTKeys
		RGB
		NonRed
		Luminance
		Alpha
		Hue
	End Enum

#End If


    <Flags()> _
    Public Enum TimelineMajorType
        None = 0
        Composite = 1
        Effect = &H10
        Group = &H80
        Source = 4
        Track = 2
        Transition = 8
    End Enum

    Public Enum TimelineInsertMode
        Insert = 1
        Overlay = 2
    End Enum

    <Flags()> _
    Public Enum SFNValidateFlags
        None = &H0
        Check = &H1
        Popup = &H2
        TellMe = &H4
        Replace = &H8
        UseLocal = &H10
        NoFind = &H20
        IgnoreMuted = &H40
        [End]
    End Enum

    <StructLayout(LayoutKind.Sequential)> _
    Public Class SCompFmt0
        Public nFormatId As Integer
        Public MediaType As AMMediaType
    End Class

    Public Enum ResizeFlags
        Stretch
        Crop
        PreserveAspectRatio
        PreserveAspectRatioNoLetterBox
    End Enum

    Public Enum DexterFTrackSearchFlags
        Bounding = -1
        ExactlyAt = 0
        Forwards = 1
    End Enum

    <StructLayout(LayoutKind.Sequential, Pack:=4)> _
    Public Structure DexterParam
        <MarshalAs(UnmanagedType.BStr)> _
        Public Name As String
        Public dispID As Integer
        Public nValues As Integer
    End Structure

    Public Enum ConnectFDynamic
        None = &H0
        Sources = &H1
        Effects = &H2
    End Enum

    <StructLayout(LayoutKind.Sequential, Pack:=8)> _
    Public Structure DexterValue
        <MarshalAs(UnmanagedType.Struct)> _
        Public v As Object
        Public rt As Long
        Public dwInterp As Dexterf
    End Structure

    Public Enum Dexterf
        Jump
        Interpolate
    End Enum

    Public Enum DESErrorCode
        BadSourceName = 1400
        BadSourceName2 = 1401
        MissingSourceName = 1402
        UnknownSource = 1403
        InstallProblem = 1404
        NoSourceNames = 1405
        BadMediaType = 1406
        StreamNumber = 1407
        OutOfMemory = 1408
        DIBSeqNotAllSame = 1409
        ClipTooShort = 1410
        InvalidDXT = 1411
        InvalidDefaultDXTT = 1412
        No3D = 1413
        BrokenDXT = 1414
        NoSuchProperty = 1415
        IllegalPropertyVal = 1416
        InvalidXML = 1417
        CantFindFilter = 1418
        DiskWriteError = 1419
        InvalidAudioFX = 1420
        CantFindCompressor = 1421
        TimelineParse = 1426
        GraphError = 1427
        GridError = 1428
        InterfaceError = 1429
    End Enum

#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("E31FB81B-1335-11D1-8189-0000F87557DB")> _
	Public Interface IDXEffect
		<PreserveSig> _
		Function get_Capabilities(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function get_Progress(ByRef pVal As Single) As Integer

		<PreserveSig> _
		Function put_Progress(newVal As Single) As Integer

		<PreserveSig> _
		Function get_StepResolution(ByRef pVal As Single) As Integer

		<PreserveSig> _
		Function get_Duration(ByRef pVal As Single) As Integer

		<PreserveSig> _
		Function put_Duration(newVal As Single) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("4EE9EAD9-DA4D-43D0-9383-06B90C08B12B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDxtAlphaSetter
		Inherits IDXEffect

        <PreserveSig()> _
        Shadows Function get_Capabilities(ByRef pVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_Progress(ByRef pVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function put_Progress(ByVal newVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function get_StepResolution(ByRef pVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function get_Duration(ByRef pVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function put_Duration(ByVal newVal As Single) As Integer

		<PreserveSig> _
		Function get_Alpha(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_Alpha(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_AlphaRamp(ByRef pVal As Double) As Integer

		<PreserveSig> _
		Function put_AlphaRamp(newVal As Double) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("BB44391E-6ABD-422F-9E2E-385C9DFF51FC"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDxtCompositor
		Inherits IDXEffect

        <PreserveSig()> _
        Shadows Function get_Capabilities(ByRef pVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_Progress(ByRef pVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function put_Progress(ByVal newVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function get_StepResolution(ByRef pVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function get_Duration(ByRef pVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function put_Duration(ByVal newVal As Single) As Integer

		<PreserveSig> _
		Function get_OffsetX(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_OffsetX(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_OffsetY(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_OffsetY(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_Width(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_Width(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_Height(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_Height(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_SrcOffsetX(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_SrcOffsetX(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_SrcOffsetY(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_SrcOffsetY(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_SrcWidth(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_SrcWidth(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_SrcHeight(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_SrcHeight(newVal As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("DE75D011-7A65-11D2-8CEA-00A0C9441E20"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDxtJpeg
		Inherits IDXEffect

        <PreserveSig()> _
        Shadows Function get_Capabilities(ByRef pVal As Integer) As Integer

        <PreserveSig()> _
        Shadows Function get_Progress(ByRef pVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function put_Progress(ByVal newVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function get_StepResolution(ByRef pVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function get_Duration(ByRef pVal As Single) As Integer

        <PreserveSig()> _
        Shadows Function put_Duration(ByVal newVal As Single) As Integer

		<PreserveSig> _
		Function get_MaskNum(ByRef MIDL_0018 As Integer) As Integer

		<PreserveSig> _
		Function put_MaskNum(MIDL_0019 As Integer) As Integer

		<PreserveSig> _
		Function get_MaskName(<MarshalAs(UnmanagedType.BStr)> ByRef pVal As String) As Integer

		<PreserveSig> _
		Function put_MaskName(<MarshalAs(UnmanagedType.BStr)> newVal As String) As Integer

		<PreserveSig> _
		Function get_ScaleX(ByRef MIDL_0020 As Double) As Integer

		<PreserveSig> _
		Function put_ScaleX(MIDL_0021 As Double) As Integer

		<PreserveSig> _
		Function get_ScaleY(ByRef MIDL_0022 As Double) As Integer

		<PreserveSig> _
		Function put_ScaleY(MIDL_0023 As Double) As Integer

		<PreserveSig> _
		Function get_OffsetX(ByRef MIDL_0024 As Integer) As Integer

		<PreserveSig> _
		Function put_OffsetX(MIDL_0025 As Integer) As Integer

		<PreserveSig> _
		Function get_OffsetY(ByRef MIDL_0026 As Integer) As Integer

		<PreserveSig> _
		Function put_OffsetY(MIDL_0027 As Integer) As Integer

		<PreserveSig> _
		Function get_ReplicateX(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_ReplicateX(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_ReplicateY(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_ReplicateY(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_BorderColor(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_BorderColor(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_BorderWidth(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_BorderWidth(newVal As Integer) As Integer

		<PreserveSig> _
		Function get_BorderSoftness(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Function put_BorderSoftness(newVal As Integer) As Integer

		<PreserveSig> _
		Function ApplyChanges() As Integer

		<PreserveSig> _
		Function LoadDefSettings() As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, Guid("3255DE56-38FB-4901-B980-94B438010D7B"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
	Public Interface IDxtKey
		Inherits IDXEffect

		<PreserveSig> _
		Shadows Function get_Capabilities(ByRef pVal As Integer) As Integer

		<PreserveSig> _
		Shadows Function get_Progress(ByRef pVal As Single) As Integer

		<PreserveSig> _
		Shadows Function put_Progress(newVal As Single) As Integer

		<PreserveSig> _
		Shadows Function get_StepResolution(ByRef pVal As Single) As Integer

		<PreserveSig> _
		Shadows Function get_Duration(ByRef pVal As Single) As Integer

		<PreserveSig> _
		Shadows Function put_Duration(newVal As Single) As Integer

		<PreserveSig> _
		Function get_KeyType(ByRef MIDL_0028 As Integer) As Integer

		<PreserveSig> _
		Function put_KeyType(MIDL_0029 As Integer) As Integer

		<PreserveSig> _
		Function get_Hue(ByRef MIDL_0030 As Integer) As Integer

		<PreserveSig> _
		Function put_Hue(MIDL_0031 As Integer) As Integer

		<PreserveSig> _
		Function get_Luminance(ByRef MIDL_0032 As Integer) As Integer

		<PreserveSig> _
		Function put_Luminance(MIDL_0033 As Integer) As Integer

		<PreserveSig> _
		Function get_RGB(ByRef MIDL_0034 As Integer) As Integer

		<PreserveSig> _
		Function put_RGB(MIDL_0035 As Integer) As Integer

		<PreserveSig> _
		Function get_Similarity(ByRef MIDL_0036 As Integer) As Integer

		<PreserveSig> _
		Function put_Similarity(MIDL_0037 As Integer) As Integer

		<PreserveSig> _
		Function get_Invert(<MarshalAs(UnmanagedType.Bool)> ByRef MIDL_0038 As Boolean) As Integer

		<PreserveSig> _
		Function put_Invert(<MarshalAs(UnmanagedType.Bool)> MIDL_0039 As Boolean) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("F03FA8DE-879A-4D59-9B2C-26BB1CF83461")> _
	Public Interface IFindCompressorCB
		<PreserveSig> _
		Function GetCompressor(<MarshalAs(UnmanagedType.LPStruct)> pType As AMMediaType, <MarshalAs(UnmanagedType.LPStruct)> pCompType As AMMediaType, ByRef ppFilter As IBaseFilter) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsDual), Guid("AE9472BE-B0C3-11D2-8D24-00A0C9441E20")> _
	Public Interface IGrfCache
		<PreserveSig> _
		Function AddFilter(ChainedCache As IGrfCache, Id As Long, pFilter As IBaseFilter, <MarshalAs(UnmanagedType.LPWStr)> pName As String) As Integer

		<PreserveSig> _
		Function ConnectPins(ChainedCache As IGrfCache, PinID1 As Long, pPin1 As IPin, PinID2 As Long, pPin2 As IPin) As Integer

		<PreserveSig> _
		Function SetGraph(pGraph As IGraphBuilder) As Integer

		<PreserveSig> _
		Function DoConnectionsNow() As Integer
	End Interface

#End If

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("E43E73A2-0EFA-11D3-9601-00A0C9441E20")> _
    Public Interface IAMErrorLog
        <PreserveSig()> _
        Function LogError(ByVal Severity As Integer, <MarshalAs(UnmanagedType.BStr)> ByVal pErrorString As String, ByVal ErrorCode As Integer, ByVal hresult As Integer, <[In]()> ByVal pExtraInfo As IntPtr) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("963566DA-BE21-4EAF-88E9-35704F8F52A1")> _
    Public Interface IAMSetErrorLog
        <PreserveSig()> _
        Function get_ErrorLog(ByRef pVal As IAMErrorLog) As Integer

        <PreserveSig()> _
        Function put_ErrorLog(ByVal newVal As IAMErrorLog) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("78530B74-61F9-11D2-8CAD-00A024580902")> _
    Public Interface IAMTimeline
        <PreserveSig()> _
        Function CreateEmptyNode(ByRef ppObj As IAMTimelineObj, ByVal Type As TimelineMajorType) As Integer

        <PreserveSig()> _
        Function AddGroup(ByVal pGroup As IAMTimelineObj) As Integer

        <PreserveSig()> _
        Function RemGroupFromList(ByVal pGroup As IAMTimelineObj) As Integer

        <PreserveSig()> _
        Function GetGroup(ByRef ppGroup As IAMTimelineObj, ByVal WhichGroup As Integer) As Integer

        <PreserveSig()> _
        Function GetGroupCount(ByRef pCount As Integer) As Integer

        <PreserveSig()> _
        Function ClearAllGroups() As Integer

        <PreserveSig()> _
        Function GetInsertMode(ByRef pMode As TimelineInsertMode) As Integer

        <PreserveSig()> _
        Function SetInsertMode(ByVal Mode As TimelineInsertMode) As Integer

        <PreserveSig()> _
        Function EnableTransitions(<MarshalAs(UnmanagedType.Bool)> ByVal fEnabled As Boolean) As Integer

        <PreserveSig()> _
        Function TransitionsEnabled(<MarshalAs(UnmanagedType.Bool)> ByRef pfEnabled As Boolean) As Integer

        <PreserveSig()> _
        Function EnableEffects(<MarshalAs(UnmanagedType.Bool)> ByVal fEnabled As Boolean) As Integer

        <PreserveSig()> _
        Function EffectsEnabled(<MarshalAs(UnmanagedType.Bool)> ByRef pfEnabled As Boolean) As Integer

        <PreserveSig()> _
        Function SetInterestRange(ByVal Start As Long, ByVal [Stop] As Long) As Integer

        <PreserveSig()> _
        Function GetDuration(ByRef pDuration As Long) As Integer

        <PreserveSig()> _
        Function GetDuration2(ByRef pDuration As Double) As Integer

        <PreserveSig()> _
        Function SetDefaultFPS(ByVal FPS As Double) As Integer

        <PreserveSig()> _
        Function GetDefaultFPS(ByRef pFPS As Double) As Integer

        <PreserveSig()> _
        Function IsDirty(<MarshalAs(UnmanagedType.Bool)> ByRef pDirty As Boolean) As Integer

        <PreserveSig()> _
        Function GetDirtyRange(ByRef pStart As Long, ByRef pStop As Long) As Integer

        <PreserveSig()> _
        Function GetCountOfType(ByVal Group As Integer, ByRef pVal As Integer, ByRef pValWithComps As Integer, ByVal majortype As TimelineMajorType) As Integer

        <PreserveSig()> _
        Function ValidateSourceNames(ByVal ValidateFlags As SFNValidateFlags, ByVal pOverride As IMediaLocator, ByVal NotifyEventHandle As IntPtr) As Integer

        <PreserveSig()> _
        Function SetDefaultTransition(<MarshalAs(UnmanagedType.LPStruct)> ByVal pGuid As Guid) As Integer

        <PreserveSig()> _
        Function GetDefaultTransition(ByRef pGuid As Guid) As Integer

        <PreserveSig()> _
        Function SetDefaultEffect(<MarshalAs(UnmanagedType.LPStruct)> ByVal pGuid As Guid) As Integer

        <PreserveSig()> _
        Function GetDefaultEffect(ByRef pGuid As Guid) As Integer

        <PreserveSig()> _
        Function SetDefaultTransitionB(<MarshalAs(UnmanagedType.BStr)> ByVal pGuid As String) As Integer

        <PreserveSig()> _
        Function GetDefaultTransitionB(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef sGuid As String) As Integer

        <PreserveSig()> _
        Function SetDefaultEffectB(<MarshalAs(UnmanagedType.BStr)> ByVal pGuid As String) As Integer

        <PreserveSig()> _
        Function GetDefaultEffectB(<Out(), MarshalAs(UnmanagedType.BStr)> ByRef sGuid As String) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("EAE58536-622E-11D2-8CAD-00A024580902")> _
    Public Interface IAMTimelineComp
        <PreserveSig()> _
        Function VTrackInsBefore(ByVal pVirtualTrack As IAMTimelineObj, ByVal priority As Integer) As Integer

        <PreserveSig()> _
        Function VTrackSwapPriorities(ByVal VirtualTrackA As Integer, ByVal VirtualTrackB As Integer) As Integer

        <PreserveSig()> _
        Function VTrackGetCount(ByRef pVal As Integer) As Integer

        <PreserveSig()> _
        Function GetVTrack(ByRef ppVirtualTrack As IAMTimelineObj, ByVal Which As Integer) As Integer

        <PreserveSig()> _
        Function GetCountOfType(ByRef pVal As Integer, ByRef pValWithComps As Integer, ByVal majortype As TimelineMajorType) As Integer

        <PreserveSig()> _
        Function GetRecursiveLayerOfType(ByRef ppVirtualTrack As IAMTimelineObj, ByVal WhichLayer As Integer, ByVal Type As TimelineMajorType) As Integer

        <PreserveSig()> _
        Function GetRecursiveLayerOfTypeI(ByRef ppVirtualTrack As IAMTimelineObj, <[In](), Out()> ByRef pWhichLayer As Integer, ByVal Type As TimelineMajorType) As Integer

        <PreserveSig()> _
        Function GetNextVTrack(ByVal pVirtualTrack As IAMTimelineObj, ByRef ppNextVirtualTrack As IAMTimelineObj) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("BCE0C264-622D-11D2-8CAD-00A024580902")> _
    Public Interface IAMTimelineEffect
        <PreserveSig()> _
        Function EffectGetPriority(ByRef pVal As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("EAE58537-622E-11D2-8CAD-00A024580902"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IAMTimelineEffectable
        <PreserveSig()> _
        Function EffectInsBefore(ByVal pFX As IAMTimelineObj, ByVal priority As Integer) As Integer

        <PreserveSig()> _
        Function EffectSwapPriorities(ByVal PriorityA As Integer, ByVal PriorityB As Integer) As Integer

        <PreserveSig()> _
        Function EffectGetCount(ByRef pCount As Integer) As Integer

        <PreserveSig()> _
        Function GetEffect(ByRef ppFx As IAMTimelineObj, ByVal Which As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("9EED4F00-B8A6-11D2-8023-00C0DF10D434"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IAMTimelineGroup
        <PreserveSig()> _
        Function SetTimeline(ByVal pTimeline As IAMTimeline) As Integer

        <PreserveSig()> _
        Function GetTimeline(ByRef ppTimeline As IAMTimeline) As Integer

        <PreserveSig()> _
        Function GetPriority(ByRef pPriority As Integer) As Integer

        <PreserveSig()> _
        Function GetMediaType(<Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

        <PreserveSig()> _
        Function SetMediaType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

        <PreserveSig()> _
        Function SetOutputFPS(ByVal FPS As Double) As Integer

        <PreserveSig()> _
        Function GetOutputFPS(ByRef pFPS As Double) As Integer

        <PreserveSig()> _
        Function SetGroupName(<MarshalAs(UnmanagedType.BStr)> ByVal pGroupName As String) As Integer

        <PreserveSig()> _
        Function GetGroupName(<MarshalAs(UnmanagedType.BStr)> ByRef pGroupName As String) As Integer

        <PreserveSig()> _
        Function SetPreviewMode(<MarshalAs(UnmanagedType.Bool)> ByVal fPreview As Boolean) As Integer

        <PreserveSig()> _
        Function GetPreviewMode(<MarshalAs(UnmanagedType.Bool)> ByRef pfPreview As Boolean) As Integer

        <PreserveSig()> _
        Function SetMediaTypeForVB(<[In]()> ByVal Val As Integer) As Integer

        <PreserveSig()> _
        Function GetOutputBuffering(ByRef pnBuffer As Integer) As Integer

        <PreserveSig()> _
        Function SetOutputBuffering(<[In]()> ByVal nBuffer As Integer) As Integer

        <PreserveSig()> _
        Function SetSmartRecompressFormat(ByVal pFormat As SCompFmt0) As Integer

        <PreserveSig()> _
        Function GetSmartRecompressFormat(ByRef ppFormat As SCompFmt0) As Integer

        <PreserveSig()> _
        Function IsSmartRecompressFormatSet(<MarshalAs(UnmanagedType.Bool)> ByRef pVal As Boolean) As Integer

        <PreserveSig()> _
        Function IsRecompressFormatDirty(<MarshalAs(UnmanagedType.Bool)> ByRef pVal As Boolean) As Integer

        <PreserveSig()> _
        Function ClearRecompressFormatDirty() As Integer

        <PreserveSig()> _
        Function SetRecompFormatFromSource(ByVal pSource As IAMTimelineSrc) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("78530B77-61F9-11D2-8CAD-00A024580902"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IAMTimelineObj
        <PreserveSig()> _
        Function GetStartStop(ByRef pStart As Long, ByRef pStop As Long) As Integer

        <PreserveSig()> _
        Function GetStartStop2(ByRef pStart As Double, ByRef pStop As Double) As Integer

        <PreserveSig()> _
        Function FixTimes(ByRef pStart As Long, ByRef pStop As Long) As Integer

        <PreserveSig()> _
        Function FixTimes2(ByRef pStart As Double, ByRef pStop As Double) As Integer

        <PreserveSig()> _
        Function SetStartStop(ByVal Start As Long, ByVal [Stop] As Long) As Integer

        <PreserveSig()> _
        Function SetStartStop2(ByVal Start As Double, ByVal [Stop] As Double) As Integer

        <PreserveSig()> _
        Function GetPropertySetter(ByRef pVal As IPropertySetter) As Integer

        <PreserveSig()> _
        Function SetPropertySetter(ByVal newVal As IPropertySetter) As Integer

        <PreserveSig()> _
        Function GetSubObject(<MarshalAs(UnmanagedType.IUnknown)> ByRef pVal As Object) As Integer

        <PreserveSig()> _
        Function SetSubObject(<[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal newVal As Object) As Integer

        <PreserveSig()> _
        Function SetSubObjectGUID(ByVal newVal As Guid) As Integer

        <PreserveSig()> _
        Function SetSubObjectGUIDB(<MarshalAs(UnmanagedType.BStr)> ByVal newVal As String) As Integer

        <PreserveSig()> _
        Function GetSubObjectGUID(ByRef pVal As Guid) As Integer

        <PreserveSig()> _
        Function GetSubObjectGUIDB(<MarshalAs(UnmanagedType.BStr)> ByRef pVal As String) As Integer

        <PreserveSig()> _
        Function GetSubObjectLoaded(<MarshalAs(UnmanagedType.Bool)> ByRef pVal As Boolean) As Integer

        <PreserveSig()> _
        Function GetTimelineType(ByRef pVal As TimelineMajorType) As Integer

        <PreserveSig()> _
        Function SetTimelineType(ByVal newVal As TimelineMajorType) As Integer

        <PreserveSig()> _
        Function GetUserID(ByRef pVal As Integer) As Integer

        <PreserveSig()> _
        Function SetUserID(ByVal newVal As Integer) As Integer

        <PreserveSig()> _
        Function GetGenID(ByRef pVal As Integer) As Integer

        <PreserveSig()> _
        Function GetUserName(<MarshalAs(UnmanagedType.BStr)> ByRef pVal As String) As Integer

        <PreserveSig()> _
        Function SetUserName(<MarshalAs(UnmanagedType.BStr)> ByVal newVal As String) As Integer

        <PreserveSig()> _
        Function GetUserData(ByVal pData As IntPtr, ByRef pSize As Integer) As Integer

        <PreserveSig()> _
        Function SetUserData(ByVal pData As IntPtr, ByVal Size As Integer) As Integer

        <PreserveSig()> _
        Function GetMuted(<MarshalAs(UnmanagedType.Bool)> ByRef pVal As Boolean) As Integer

        <PreserveSig()> _
        Function SetMuted(<MarshalAs(UnmanagedType.Bool)> ByVal newVal As Boolean) As Integer

        <PreserveSig()> _
        Function GetLocked(<MarshalAs(UnmanagedType.Bool)> ByRef pVal As Boolean) As Integer

        <PreserveSig()> _
        Function SetLocked(<MarshalAs(UnmanagedType.Bool)> ByVal newVal As Boolean) As Integer

        <PreserveSig()> _
        Function GetDirtyRange(ByRef pStart As Long, ByRef pStop As Long) As Integer

        <PreserveSig()> _
        Function GetDirtyRange2(ByRef pStart As Double, ByRef pStop As Double) As Integer

        <PreserveSig()> _
        Function SetDirtyRange(ByVal Start As Long, ByVal [Stop] As Long) As Integer

        <PreserveSig()> _
        Function SetDirtyRange2(ByVal Start As Double, ByVal [Stop] As Double) As Integer

        <PreserveSig()> _
        Function ClearDirty() As Integer

        <PreserveSig()> _
        Function Remove() As Integer

        <PreserveSig()> _
        Function RemoveAll() As Integer

        <PreserveSig()> _
        Function GetTimelineNoRef(ByRef ppResult As IAMTimeline) As Integer

        <PreserveSig()> _
        Function GetGroupIBelongTo(ByRef ppGroup As IAMTimelineGroup) As Integer

        <PreserveSig()> _
        Function GetEmbedDepth(ByRef pVal As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("A0F840A0-D590-11D2-8D55-00A0C9441E20")> _
    Public Interface IAMTimelineSplittable
        <PreserveSig()> _
        Function SplitAt(ByVal Time As Long) As Integer

        <PreserveSig()> _
        Function SplitAt2(ByVal Time As Double) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("78530B79-61F9-11D2-8CAD-00A024580902"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IAMTimelineSrc
        <PreserveSig()> _
        Function GetMediaTimes(ByRef pStart As Long, ByRef pStop As Long) As Integer

        <PreserveSig()> _
        Function GetMediaTimes2(ByRef pStart As Double, ByRef pStop As Double) As Integer

        <PreserveSig()> _
        Function ModifyStopTime(ByVal [Stop] As Long) As Integer

        <PreserveSig()> _
        Function ModifyStopTime2(ByVal [Stop] As Double) As Integer

        <PreserveSig()> _
        Function FixMediaTimes(ByRef pStart As Long, ByRef pStop As Long) As Integer

        <PreserveSig()> _
        Function FixMediaTimes2(ByRef pStart As Double, ByRef pStop As Double) As Integer

        <PreserveSig()> _
        Function SetMediaTimes(ByVal Start As Long, ByVal [Stop] As Long) As Integer

        <PreserveSig()> _
        Function SetMediaTimes2(ByVal Start As Double, ByVal [Stop] As Double) As Integer

        <PreserveSig()> _
        Function SetMediaLength(ByVal Length As Long) As Integer

        <PreserveSig()> _
        Function SetMediaLength2(ByVal Length As Double) As Integer

        <PreserveSig()> _
        Function GetMediaLength(ByRef pLength As Long) As Integer

        <PreserveSig()> _
        Function GetMediaLength2(ByRef pLength As Double) As Integer

        <PreserveSig()> _
        Function GetMediaName(<MarshalAs(UnmanagedType.BStr)> ByRef pVal As String) As Integer

        <PreserveSig()> _
        Function SetMediaName(<MarshalAs(UnmanagedType.BStr)> ByVal newVal As String) As Integer

        <PreserveSig()> _
        Function SpliceWithNext(ByVal pNext As IAMTimelineObj) As Integer

        <PreserveSig()> _
        Function GetStreamNumber(ByRef pVal As Integer) As Integer

        <PreserveSig()> _
        Function SetStreamNumber(ByVal Val As Integer) As Integer

        <PreserveSig()> _
        Function IsNormalRate(<MarshalAs(UnmanagedType.Bool)> ByRef pVal As Boolean) As Integer

        <PreserveSig()> _
        Function GetDefaultFPS(ByRef pFPS As Double) As Integer

        <PreserveSig()> _
        Function SetDefaultFPS(ByVal FPS As Double) As Integer

        <PreserveSig()> _
        Function GetStretchMode(ByRef pnStretchMode As ResizeFlags) As Integer

        <PreserveSig()> _
        Function SetStretchMode(ByVal nStretchMode As ResizeFlags) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("EAE58538-622E-11D2-8CAD-00A024580902"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IAMTimelineTrack
        <PreserveSig()> _
        Function SrcAdd(ByVal pSource As IAMTimelineObj) As Integer

        <PreserveSig()> _
        Function GetNextSrc(ByRef ppSrc As IAMTimelineObj, ByRef pInOut As Long) As Integer

        <PreserveSig()> _
        Function GetNextSrc2(ByRef ppSrc As IAMTimelineObj, ByRef pInOut As Double) As Integer

        <PreserveSig()> _
        Function MoveEverythingBy(ByVal Start As Long, ByVal MoveBy As Long) As Integer

        <PreserveSig()> _
        Function MoveEverythingBy2(ByVal Start As Double, ByVal MoveBy As Double) As Integer

        <PreserveSig()> _
        Function GetSourcesCount(ByRef pVal As Integer) As Integer

        <PreserveSig()> _
        Function AreYouBlank(<MarshalAs(UnmanagedType.Bool)> ByRef pVal As Boolean) As Integer

        <PreserveSig()> _
        Function GetSrcAtTime(ByRef ppSrc As IAMTimelineObj, ByVal Time As Long, ByVal SearchDirection As DexterFTrackSearchFlags) As Integer

        <PreserveSig()> _
        Function GetSrcAtTime2(ByRef ppSrc As IAMTimelineObj, ByVal Time As Double, ByVal SearchDirection As DexterFTrackSearchFlags) As Integer

        <PreserveSig()> _
        Function InsertSpace(ByVal rtStart As Long, ByVal rtEnd As Long) As Integer

        <PreserveSig()> _
        Function InsertSpace2(ByVal rtStart As Double, ByVal rtEnd As Double) As Integer

        <PreserveSig()> _
        Function ZeroBetween(ByVal rtStart As Long, ByVal rtEnd As Long) As Integer

        <PreserveSig()> _
        Function ZeroBetween2(ByVal rtStart As Double, ByVal rtEnd As Double) As Integer

        <PreserveSig()> _
        Function GetNextSrcEx(ByVal pLast As IAMTimelineObj, ByRef ppNext As IAMTimelineObj) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("BCE0C265-622D-11D2-8CAD-00A024580902")> _
    Public Interface IAMTimelineTrans
        <PreserveSig()> _
        Function GetCutPoint(ByRef pTLTime As Long) As Integer

        <PreserveSig()> _
        Function GetCutPoint2(ByRef pTLTime As Double) As Integer

        <PreserveSig()> _
        Function SetCutPoint(ByVal TLTime As Long) As Integer

        <PreserveSig()> _
        Function SetCutPoint2(ByVal TLTime As Double) As Integer

        <PreserveSig()> _
        Function GetSwapInputs(<MarshalAs(UnmanagedType.Bool)> ByRef pVal As Boolean) As Integer

        <PreserveSig()> _
        Function SetSwapInputs(<MarshalAs(UnmanagedType.Bool)> ByVal pVal As Boolean) As Integer

        <PreserveSig()> _
        Function GetCutsOnly(<MarshalAs(UnmanagedType.Bool)> ByRef pVal As Boolean) As Integer

        <PreserveSig()> _
        Function SetCutsOnly(<MarshalAs(UnmanagedType.Bool)> ByVal pVal As Boolean) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("378FA386-622E-11D2-8CAD-00A024580902"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IAMTimelineTransable
        <PreserveSig()> _
        Function TransAdd(ByVal pTrans As IAMTimelineObj) As Integer

        <PreserveSig()> _
        Function TransGetCount(ByRef pCount As Integer) As Integer

        <PreserveSig()> _
        Function GetNextTrans(ByRef ppTrans As IAMTimelineObj, ByRef pInOut As Long) As Integer

        <PreserveSig()> _
        Function GetNextTrans2(ByRef ppTrans As IAMTimelineObj, ByRef pInOut As Double) As Integer

        <PreserveSig()> _
        Function GetTransAtTime(ByRef ppObj As IAMTimelineObj, ByVal Time As Long, ByVal SearchDirection As DexterFTrackSearchFlags) As Integer

        <PreserveSig()> _
        Function GetTransAtTime2(ByRef ppObj As IAMTimelineObj, ByVal Time As Double, ByVal SearchDirection As DexterFTrackSearchFlags) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("A8ED5F80-C2C7-11D2-8D39-00A0C9441E20"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface IAMTimelineVirtualTrack
        <PreserveSig()> _
        Function TrackGetPriority(ByRef pPriority As Integer) As Integer

        <PreserveSig()> _
        Function SetTrackDirty() As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("65BD0710-24D2-4ff7-9324-ED2E5D3ABAFA")> _
    Public Interface IMediaDet
        <PreserveSig()> _
        Function get_Filter(<MarshalAs(UnmanagedType.IUnknown)> ByRef pVal As Object) As Integer

        <PreserveSig()> _
        Function put_Filter(<MarshalAs(UnmanagedType.IUnknown)> ByVal newVal As Object) As Integer

        <PreserveSig()> _
        Function get_OutputStreams(ByRef pVal As Integer) As Integer

        <PreserveSig()> _
        Function get_CurrentStream(ByRef pVal As Integer) As Integer

        <PreserveSig()> _
        Function put_CurrentStream(ByVal newVal As Integer) As Integer

        <PreserveSig()> _
        Function get_StreamType(ByRef pVal As Guid) As Integer

        <PreserveSig()> _
        Function get_StreamTypeB(<MarshalAs(UnmanagedType.BStr)> ByRef pVal As String) As Integer

        <PreserveSig()> _
        Function get_StreamLength(ByRef pVal As Double) As Integer

        <PreserveSig()> _
        Function get_Filename(<MarshalAs(UnmanagedType.BStr)> ByRef pVal As String) As Integer

        <PreserveSig()> _
        Function put_Filename(<MarshalAs(UnmanagedType.BStr)> ByVal newVal As String) As Integer

        <PreserveSig()> _
        Function GetBitmapBits(ByVal StreamTime As Double, ByRef pBufferSize As Integer, <[In]()> ByVal pBuffer As IntPtr, ByVal Width As Integer, ByVal Height As Integer) As Integer

        <PreserveSig()> _
        Function WriteBitmapBits(ByVal StreamTime As Double, ByVal Width As Integer, ByVal Height As Integer, <[In](), MarshalAs(UnmanagedType.BStr)> ByVal Filename As String) As Integer

        <PreserveSig()> _
        Function get_StreamMediaType(<Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pVal As AMMediaType) As Integer

        <PreserveSig()> _
        Function GetSampleGrabber(ByRef ppVal As ISampleGrabber) As Integer

        <PreserveSig()> _
        Function get_FrameRate(ByRef pVal As Double) As Integer

        <PreserveSig()> _
        Function EnterBitmapGrabMode(ByVal SeekTime As Double) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("288581E0-66CE-11D2-918F-00C0DF10D434")> _
    Public Interface IMediaLocator
        <PreserveSig()> _
        Function FindMediaFile(<MarshalAs(UnmanagedType.BStr)> ByVal Input As String, <MarshalAs(UnmanagedType.BStr)> ByVal FilterString As String, <MarshalAs(UnmanagedType.BStr)> ByRef pOutput As String, ByVal Flags As SFNValidateFlags) As Integer

        <PreserveSig()> _
        Function AddFoundLocation(<MarshalAs(UnmanagedType.BStr)> ByVal DirectoryName As String) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("AE9472BD-B0C3-11D2-8D24-00A0C9441E20")> _
    Public Interface IPropertySetter
        <PreserveSig()> _
        Function LoadXML(<[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pxml As Object) As Integer

        <PreserveSig()> _
        Function PrintXML(<Out(), MarshalAs(UnmanagedType.LPStr)> ByVal pszXML As StringBuilder, <[In]()> ByVal cbXML As Integer, ByRef pcbPrinted As Integer, <[In]()> ByVal indent As Integer) As Integer

        <PreserveSig()> _
        Function CloneProps(ByRef ppSetter As IPropertySetter, <[In]()> ByVal rtStart As Long, <[In]()> ByVal rtStop As Long) As Integer

        <PreserveSig()> _
        Function AddProp(<[In]()> ByVal Param As DexterParam, <[In](), MarshalAs(UnmanagedType.LPArray)> ByVal paValue As DexterValue()) As Integer

        <PreserveSig()> _
        Function GetProps(ByRef pcParams As Integer, ByRef paParam As IntPtr, ByRef paValue As IntPtr) As Integer

        <PreserveSig()> _
        Function FreeProps(<[In]()> ByVal cParams As Integer, <[In]()> ByVal paParam As IntPtr, <[In]()> ByVal paValue As IntPtr) As Integer

        <PreserveSig()> _
        Function ClearProps() As Integer

        <PreserveSig()> _
        Function SaveToBlob(ByRef pcSize As Integer, ByRef ppb As IntPtr) As Integer

        <PreserveSig()> _
        Function LoadFromBlob(<[In]()> ByVal cSize As Integer, <[In]()> ByVal pb As IntPtr) As Integer

        <PreserveSig()> _
        Function SetProps(<[In](), MarshalAs(UnmanagedType.IUnknown)> ByVal pTarget As Object, <[In]()> ByVal rtNow As Long) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("6BEE3A81-66C9-11D2-918F-00C0DF10D434")> _
    Public Interface IRenderEngine
        <PreserveSig()> _
        Function SetTimelineObject(ByVal pTimeline As IAMTimeline) As Integer

        <PreserveSig()> _
        Function GetTimelineObject(ByRef ppTimeline As IAMTimeline) As Integer

        <PreserveSig()> _
        Function GetFilterGraph(ByRef ppFG As IGraphBuilder) As Integer

        <PreserveSig()> _
        Function SetFilterGraph(ByVal pFG As IGraphBuilder) As Integer

        <PreserveSig()> _
        Function SetInterestRange(ByVal Start As Long, ByVal [Stop] As Long) As Integer

        <PreserveSig()> _
        Function SetInterestRange2(ByVal Start As Double, ByVal [Stop] As Double) As Integer

        <PreserveSig()> _
        Function SetRenderRange(ByVal Start As Long, ByVal [Stop] As Long) As Integer

        <PreserveSig()> _
        Function SetRenderRange2(ByVal Start As Double, ByVal [Stop] As Double) As Integer

        <PreserveSig()> _
        Function GetGroupOutputPin(ByVal Group As Integer, ByRef ppRenderPin As IPin) As Integer

        <PreserveSig()> _
        Function ScrapIt() As Integer

        <PreserveSig()> _
        Function RenderOutputPins() As Integer

        <PreserveSig()> _
        Function GetVendorString(<MarshalAs(UnmanagedType.BStr)> ByRef sVendor As String) As Integer

        <PreserveSig()> _
        Function ConnectFrontEnd() As Integer

#If ALLOW_UNTESTED_INTERFACES Then
		<PreserveSig> _
		Function SetSourceConnectCallback(pCallback As IGrfCache) As Integer
#Else
        <PreserveSig()> _
        Function SetSourceConnectCallback(ByVal pCallback As Object) As Integer
#End If

        <PreserveSig()> _
        Function SetDynamicReconnectLevel(ByVal Level As ConnectFDynamic) As Integer

        <PreserveSig()> _
        Function DoSmartRecompression() As Integer

        <PreserveSig()> _
        Function UseInSmartRecompressionGraph() As Integer

        <PreserveSig()> _
        Function SetSourceNameValidation(<MarshalAs(UnmanagedType.BStr)> ByVal FilterString As String, ByVal pOverride As IMediaLocator, ByVal Flags As SFNValidateFlags) As Integer

        <PreserveSig()> _
        Function Commit() As Integer

        <PreserveSig()> _
        Function Decommit() As Integer

        <PreserveSig()> _
        Function GetCaps(ByVal Index As Integer, ByRef pReturn As Integer) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), Guid("F03FA8CE-879A-4D59-9B2C-26BB1CF83461"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)> _
    Public Interface ISmartRenderEngine
        <PreserveSig()> _
        Function SetGroupCompressor(ByVal Group As Integer, ByVal pCompressor As IBaseFilter) As Integer

        <PreserveSig()> _
        Function GetGroupCompressor(ByVal Group As Integer, ByRef pCompressor As IBaseFilter) As Integer

        <PreserveSig()> _
        Function SetFindCompressorCB(ByVal pCallback As Object) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsDual), Guid("18C628ED-962A-11D2-8D08-00A0C9441E20")> _
    Public Interface IXml2Dex
        <PreserveSig()> _
        Function CreateGraphFromFile(<MarshalAs(UnmanagedType.IUnknown)> ByRef ppGraph As Object, <MarshalAs(UnmanagedType.IUnknown)> ByVal pTimeline As Object, <MarshalAs(UnmanagedType.BStr)> ByVal Filename As String) As Integer

        <PreserveSig()> _
        Function WriteGrfFile(<MarshalAs(UnmanagedType.IUnknown)> ByVal pGraph As Object, <MarshalAs(UnmanagedType.BStr)> ByVal Filename As String) As Integer

        <PreserveSig()> _
        Function WriteXMLFile(<MarshalAs(UnmanagedType.IUnknown)> ByVal pTimeline As Object, <MarshalAs(UnmanagedType.BStr)> ByVal Filename As String) As Integer

        <PreserveSig()> _
        Function ReadXMLFile(<MarshalAs(UnmanagedType.IUnknown)> ByVal pTimeline As Object, <MarshalAs(UnmanagedType.BStr)> ByVal XMLName As String) As Integer

        <PreserveSig()> _
        Function Delete(<MarshalAs(UnmanagedType.IUnknown)> ByVal pTimeline As Object, ByVal dStart As Double, ByVal dEnd As Double) As Integer

        <PreserveSig()> _
        Function WriteXMLPart(<MarshalAs(UnmanagedType.IUnknown)> ByVal pTimeline As Object, ByVal dStart As Double, ByVal dEnd As Double, <MarshalAs(UnmanagedType.BStr)> ByVal Filename As String) As Integer

        <PreserveSig()> _
        Function PasteXMLFile(<MarshalAs(UnmanagedType.IUnknown)> ByVal pTimeline As Object, ByVal dStart As Double, <MarshalAs(UnmanagedType.BStr)> ByVal Filename As String) As Integer

        <PreserveSig()> _
        Function CopyXML(<MarshalAs(UnmanagedType.IUnknown)> ByVal pTimeline As Object, ByVal dStart As Double, ByVal dEnd As Double) As Integer

        <PreserveSig()> _
        Function PasteXML(<MarshalAs(UnmanagedType.IUnknown)> ByVal pTimeline As Object, ByVal dStart As Double) As Integer

        <PreserveSig()> _
        Function Reset() As Integer

        <PreserveSig()> _
        Function ReadXML(<MarshalAs(UnmanagedType.IUnknown)> ByVal pTimeline As Object, <MarshalAs(UnmanagedType.IUnknown)> ByVal pxml As Object) As Integer

        <PreserveSig()> _
        Function WriteXML(<MarshalAs(UnmanagedType.IUnknown)> ByVal pTimeline As Object, <MarshalAs(UnmanagedType.BStr)> ByRef pbstrXML As String) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("6BEE3A82-66C9-11d2-918F-00C0DF10D434")> _
    Public Interface IRenderEngine2
        <PreserveSig()> _
        Function SetResizerGUID(<[In]()> ByVal ResizerGuid As Guid) As Integer
    End Interface

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("4ada63a0-72d5-11d2-952a-0060081840bc")> _
    Public Interface IResize
        <PreserveSig()> _
        Function get_Size(ByRef piHeight As Integer, ByRef piWidth As Integer, ByRef pFlag As ResizeFlags) As Integer

        <PreserveSig()> _
        Function get_InputSize(ByRef piHeight As Integer, ByRef piWidth As Integer) As Integer

        <PreserveSig()> _
        Function put_Size(ByVal Height As Integer, ByVal Width As Integer, ByVal Flag As ResizeFlags) As Integer

        <PreserveSig()> _
        Function get_MediaType(<Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer

        <PreserveSig()> _
        Function put_MediaType(<[In](), MarshalAs(UnmanagedType.LPStruct)> ByVal pmt As AMMediaType) As Integer
    End Interface

End Namespace
