
import reverbsimulator as rs

import math
import numpy as np

import turtle
import matplotlib.pyplot as plt

class Animator():

    def __init__(self):
        self.turtle = turtle.Turtle()
        # 0 == fastest
        self.turtle.speed(0)

        # in pixels/meter
        self.scale = 100.0

        # RGB
        self.colors = {
            "misc": (0.2, 0.2, 0.2),
            "ray": (0.8, 0.0, 0.0),
            "material": (0.0, 0.0, 1.0),
            "speaker": (0.0, 0.8, 0.5),
            "mic": (1.0, 0.0, 0.0)
        }

    def goto(self, x, y=None):
        if y is None:
            self.turtle.goto(self.scale * x[0], self.scale * x[1])
        else:
            self.turtle.goto(self.scale * x, self.scale * y)

    def get_direction(self):
        return math.pi * self.turtle.geth() / 180.0

    def set_direction(self, direction):
        if isinstance(direction, float) or isinstance(direction, float):
            self.turtle.seth(180.0 * direction / math.pi)
        else:
            self.turtle.seth(180.0 * Point(direction).direction / math.pi)
            

    direction = property(get_direction, set_direction)


    def draw_scene(self, scn):
        def draw_line(l):
            self.turtle.up()
            self.goto(l["start"])
            self.direction = (l["end"] - l["start"]).direction
            self.turtle.down()
            self.goto(l["end"])
            self.turtle.up()

        def draw_circle(c):
            self.turtle.up()
            self.goto(c["center"].x, c["center"].y - c["radius"])
            self.direction = 0.0
            self.turtle.down()
            self.turtle.circle(self.scale * c["radius"])
            self.turtle.up()

        def draw_geom(g, col):
            self.turtle.color(col)

            if isinstance(g, rs.Line):
                draw_line(g)
            elif isinstance(g, rs.Polygon):
                #self.turtle.begin_fill()
                for i in g.lines:
                    draw_line(i)
                #self.turtle.end_fill()

            elif isinstance(g, rs.Circle):
                #self.turtle.begin_fill()
                draw_circle(g)
                #self.turtle.end_fill()

        def draw_mic(m):
            self.turtle.color(self.colors["mic"])
            self.goto(m.pos)
            self.turtle.dot(size=4)
            if m.tag is None:
                self.turtle.write("mic", align="center", font=("Arial", 14))
            else:
                self.turtle.write("mic '%s'" % m.tag, align="center", font=("Arial", 14))


        for o in scn.objs:
            if isinstance(o, rs.Geometry):
                draw_geom(o, self.colors["misc"])
            elif isinstance(o, rs.Material):
                draw_geom(o.geom, self.colors["material"])
            elif isinstance(o, rs.Speaker):
                draw_geom(o.geom, self.colors["speaker"])
            elif isinstance(o, rs.Mic):
                draw_mic(o)

    def draw_ray(self, ray, distance):
        end_draw = ray.start + rs.Point.polar(radius=distance, direction=ray.direction)
        self.turtle.up()
        self.turtle.color(self.colors["ray"])
        self.goto(ray.start)
        self.direction = ray.direction
        self.turtle.down()

        self.goto(end_draw)
        self.turtle.up()

    def stall(self):
        turtle.done()
        

def graph_FFT(audio):
    if isinstance(audio, rs.ImpulseResponse):
        graph_FFT(audio.flattened())
    else:
        plt.plot(np.fft.rfftfreq(len(audio), 1.0 / 44100.0), np.abs(np.fft.rfft(audio)))
        plt.show()

def graph_samples(audio):
    if isinstance(audio, rs.ImpulseResponse):
        graph_samples(audio.flattened())
    else:
        plt.plot(np.arange(0.0, len(audio) / 44100.0, 1.0 / 44100.0), audio)
        plt.show()

