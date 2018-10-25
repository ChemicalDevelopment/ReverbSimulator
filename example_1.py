#!/usr/bin/env python3
# reverb simulation example
import reverbsimulator as rs
import time

do_anim = False

scene = rs.Scene()

scene.add_obj(rs.Material(
    rs.Polygon((3.0, 3.0), (-3.0, 3.0), (-3.0, -3.0), (3.0, -3.0)),
    rs.ImpulseResponse.RC_lowpass(14500),
    tag="MainWall"
))
scene.add_obj(rs.Material(
    rs.Circle((0.0, 0.0), 2.0),
    rs.ImpulseResponse.RC_highpass(150) * rs.ImpulseResponse.RC_lowpass(16500),
    tag="CenterPillar"
))

scene.add_obj(rs.Speaker(
    rs.Circle((-3.0, -3.0), 0.5),
    tag="MainSpeaker"
))

scene.add_obj(rs.Mic(
    (2.5, 2.5),
    tag="MainMic"
))


anim = None

if do_anim:
    anim = rs.Animator()
    anim.draw_scene(scene)

st = time.time()
result_IR = scene.create_IR(mic_tag="MainMic", Nsamples=100, max_bounces=100, anim=anim).flattened()
print ("gen took %f s" % (time.time() - st))

#rs.graph_FFT(result_IR)

st = time.time()
rs.write_wav("impulse_example_1.wav", result_IR)
print ("output took %f s" % (time.time() - st))


#rs.graph_FFT(rs.ImpulseResponse.response_filter(lambda hz: 1.0 * hz / 22050.0, num_pts=512).flattened())
#rs.graph_FFT(rs.ImpulseResponse.RC_highpass(12000.0, 1000).flattened())

