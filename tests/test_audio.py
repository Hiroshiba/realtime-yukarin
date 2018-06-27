import pyaudio

audio_instance = pyaudio.PyAudio()

# output device
n = audio_instance.get_device_count()

for i in range(n):
    print(i, 'CABLE Input' in str(audio_instance.get_device_info_by_index(i)['name']))

print(audio_instance.get_device_info_by_index(10))
