
Imports System.Runtime.InteropServices

Namespace MultimediaStreaming

#If ALLOW_UNTESTED_INTERFACES Then

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("327FC560-AF60-11D0-8212-00C04FC32C45")> _
	Public Interface IMemoryData
		<PreserveSig> _
		Function SetBuffer(<[In]> cbSize As Integer, <[In]> pbData As IntPtr, <[In]> dwFlags As Integer) As Integer

		<PreserveSig> _
		Function GetInfo(ByRef pdwLength As Integer, <Out> ppbData As IntPtr, ByRef pcbActualData As Integer) As Integer

		<PreserveSig> _
		Function SetActual(<[In]> cbDataValid As Integer) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("54C719C0-AF60-11D0-8212-00C04FC32C45")> _
	Public Interface IAudioData
		Inherits IMemoryData

        <PreserveSig()> _
        Shadows Function SetBuffer(<[In]()> ByVal cbSize As Integer, <[In]()> ByVal pbData As IntPtr, <[In]()> ByVal dwFlags As Integer) As Integer

        <PreserveSig()> _
        Shadows Function GetInfo(ByRef pdwLength As Integer, <Out()> ByVal ppbData As IntPtr, ByRef pcbActualData As Integer) As Integer

        <PreserveSig()> _
        Shadows Function SetActual(<[In]()> ByVal cbDataValid As Integer) As Integer

		<PreserveSig> _
		Function GetFormat(<Out> pWaveFormatCurrent As WaveFormatEx) As Integer

		<PreserveSig> _
		Function SetFormat(<[In], MarshalAs(UnmanagedType.LPStruct)> lpWaveFormat As WaveFormatEx) As Integer
	End Interface

	<ComImport, System.Security.SuppressUnmanagedCodeSecurity, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("345FEE00-ABA5-11D0-8212-00C04FC32C45")> _
	Public Interface IAudioStreamSample
		Inherits IStreamSample

        <PreserveSig()> _
        Shadows Function GetMediaStream(<MarshalAs(UnmanagedType.[Interface])> ByRef ppMediaStream As IMediaStream) As Integer

        <PreserveSig()> _
        Shadows Function GetSampleTimes(ByRef pStartTime As Long, ByRef pEndTime As Long, ByRef pCurrentTime As Long) As Integer

        <PreserveSig()> _
        Shadows Function SetSampleTimes(<[In]()> ByVal pStartTime As DsLong, <[In]()> ByVal pEndTime As DsLong) As Integer

        <PreserveSig()> _
        Shadows Function Update(<[In]()> ByVal dwFlags As SSUpdate, <[In]()> ByVal hEvent As IntPtr, <[In]()> ByVal pfnAPC As IntPtr, <[In]()> ByVal dwAPCData As IntPtr) As Integer

        <PreserveSig()> _
        Shadows Function CompletionStatus(<[In]()> ByVal dwFlags As CompletionStatusFlags, <[In]()> ByVal dwMilliseconds As Integer) As Integer

		<PreserveSig> _
		Function GetAudioData(<MarshalAs(UnmanagedType.[Interface])> ByRef ppAudio As IAudioData) As Integer
	End Interface

#End If

    <ComImport(), System.Security.SuppressUnmanagedCodeSecurity(), InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("F7537560-A3BE-11D0-8212-00C04FC32C45")> _
    Public Interface IAudioMediaStream
        Inherits IMediaStream

        <PreserveSig()> _
        Shadows Function GetMultiMediaStream(<MarshalAs(UnmanagedType.[Interface])> ByRef ppMultiMediaStream As IMultiMediaStream) As Integer

        <PreserveSig()> _
        Shadows Function GetInformation(ByRef pPurposeId As Guid, ByRef pType As StreamType) As Integer

        <PreserveSig()> _
        Shadows Function SetSameFormat(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pStreamThatHasDesiredFormat As IMediaStream, <[In]()> ByVal dwFlags As Integer) As Integer

        <PreserveSig()> _
        Shadows Function AllocateSample(<[In]()> ByVal dwFlags As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppSample As IStreamSample) As Integer

        <PreserveSig()> _
        Shadows Function CreateSharedSample(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pExistingSample As IStreamSample, <[In]()> ByVal dwFlags As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppNewSample As IStreamSample) As Integer

        <PreserveSig()> _
        Shadows Function SendEndOfStream(ByVal dwFlags As Integer) As Integer

        <PreserveSig()> _
        Function GetFormat(<Out(), MarshalAs(UnmanagedType.LPStruct)> ByVal pWaveFormatCurrent As WaveFormatEx) As Integer

        <PreserveSig()> _
        Function SetFormat(<[In]()> ByVal lpWaveFormat As WaveFormatEx) As Integer

#If ALLOW_UNTESTED_INTERFACES Then
		<PreserveSig> _
		Function CreateSample(<[In], MarshalAs(UnmanagedType.[Interface])> pAudioData As IAudioData, <[In]> dwFlags As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppSample As IAudioStreamSample) As Integer
#Else
        <PreserveSig()> _
        Function CreateSample(<[In](), MarshalAs(UnmanagedType.[Interface])> ByVal pAudioData As Object, <[In]()> ByVal dwFlags As Integer, <MarshalAs(UnmanagedType.[Interface])> ByRef ppSample As Object) As Integer
#End If

    End Interface


End Namespace
