
import math
import time
import numpy as np
import scipy.io.wavfile
import scipy.signal
import multiprocessing

try:
    #import visualization submodule
    import reverbsimulator.viz
except Exception as e:
    # there was an error in the extra depends
    print ("error while importing viz: '%s'" % str(e))

def db(x, inv=False):
    if inv:
        # if inverted, return how many db a coefficient is
        if x == 0 or x == 0.0:
            return -float("inf")
        else:
            return 20.0 * math.log(x, 10.0)
    else:
        # from db to coefficient
        return math.pow(10.0, x/20.0)

def semitones(x, inv=False):
    if inv:
        # from semitones to hz
        return 440.0 * math.pow(2.0, x / 12.0)
    else:
        # from hz to semitones
        return 12.0 * math.log(x / 440.0, 2.0)

def normalize(data):
    return data/np.max(np.abs(data))

def write_wav(filename, data):
    scipy.io.wavfile.write(filename, 44100, np.int16(normalize(data) * 32767))

def read_wav(filename):
    _, src = scipy.io.wavfile.read(filename)
    return src


class ImpulseResponse():

    def __init__(self, data, delay=0.0, samplerate=44100):
        # very small epsilon to consider zero
        # TODO: find a faster way to do this
        #epsi = 0.0
        #epsi = 1.0 / (500 * (len(data) + 5))
        #if epsi > 10.0 **-16.0:
        #    epsi = 10 ** -16.0

        imp = np.array(data, dtype=np.float32)
        self.data = imp
        
        #print(np.nonzero(tmp))
        #print(np.nonzero(tmp)[-1])
        #if len(data) in (0, 1):
        #    self.data = imp
        #else:
        #    tmp = np.abs(imp) < epsi
        #    i = len(imp) - 1
        #    while i > 0 and tmp[i]:
        #        i -= 1
        #    self.data = imp[:i+1]
        self.delay = delay
        self.samplerate = samplerate

    def copy(self):
        return ImpulseResponse(self.data, self.delay, self.samplerate)


    # generative functions

    def impulse(delay=0.0, db=0.0):
        return ImpulseResponse(np.array([1.0], dtype=np.float32), delay)

    def nothing():
        return ImpulseResponse(np.array([], dtype=np.float32))

    def RC_lowpass(cutoff, num_pts=100):
        #this is an emulation of an infinite response (https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter)
        # num_pts is how many points it is let out to
        dt = 1.0 / 44100.0
        RC = 1.0 / (cutoff * 2.0 * math.pi)
        alpha =  dt / (RC + dt)

        data = [alpha * math.pow(1.0 - alpha, i) for i in range(num_pts)]

        return ImpulseResponse(data, 0.0, 44100)


    def RC_highpass(cutoff, num_pts=100):
        #this is an emulation of an infinite response (https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter)
        # num_pts is how many points it is let out to
        dt = 1.0 / 44100.0
        RC = 1.0 / (cutoff * 2.0 * math.pi)
        alpha =  dt / (RC + dt)


        data = [math.pow(-1, i) * (math.pow(alpha, i + 1) - math.pow(alpha, i)) for i in range(num_pts)]
        return ImpulseResponse(data, 0.0, 44100)

    def response_filter(func, num_pts=2048):
        # func should be a function/lambda that takes:
        # hz
        # and returns a complex number that describes the response to that frequency
        # (you can use just a float to keep phase undisturbed)

        response = np.array([func(44100.0 * i / (num_pts * 2)) for i in range(num_pts)], np.complex64)
        imp = np.fft.irfft(response)
        #print (imp)

        return ImpulseResponse(imp, 0.0, 44100)


    # class methods

    def add_delay(self, amount):
        self.delay += amount

    def combine(self, otherIR):
        if len(self.data) == 0 or len(otherIR.data) == 0:
            return ImpulseResponse([], 0.0)
        else:
            # TODO: possibly use fftpack to convolve quicker (for larger sets)
            conv_signal = None
            if len(self.data) == 1 and len(otherIR.data) == 1:
                conv_signal = [self.data[0] * otherIR.data[0]]
            else:
                conv_signal = np.convolve(self.data, otherIR.data)
            
            return ImpulseResponse(conv_signal, self.delay + otherIR.delay)

    def __add__(self, v):
        # adds them together
        if isinstance(v, ImpulseResponse):
            if self.delay > v.delay:
                _data = v.data.copy()
                _adj_my = np.concatenate([np.zeros(int((self.delay - v.delay) * self.samplerate)), self.data])
                if len(_data) > len(_adj_my):
                    _adj_my = np.concatenate([_adj_my, np.zeros(len(_data) - len(_adj_my))])
                elif len(_adj_my) > len(_data):
                    _data = np.concatenate([_data, np.zeros(len(_adj_my) - len(_data))])
                return ImpulseResponse(_data + _adj_my, v.delay, self.samplerate)
            else:
                _data = self.data.copy()
                _adj_other = np.concatenate([np.zeros(int((v.delay - self.delay) * self.samplerate)), v.data])
                if len(_data) > len(_adj_other):
                    _adj_other = np.concatenate([_adj_other, np.zeros(len(_data) - len(_adj_other))])
                elif len(_adj_other) > len(_data):
                    _data = np.concatenate([_data, np.zeros(len(_adj_other) - len(_data))])
                return ImpulseResponse(_data + _adj_other, self.delay, self.samplerate)


    def __mul__(self, v):
        if isinstance(v, ImpulseResponse):
            return self.combine(v)
        elif isinstance(v, float) or isinstance(v, int):
            r = self.copy()
            r.data *= v
            return r
    def __rmul__(self, v):
        return self.__mul__(v)

    def flattened(self):

        extra = np.zeros((int(self.delay * self.samplerate), ), dtype=np.float32)
        ir = np.concatenate([extra, self.data])

        # at the back doesn't matter
        return np.trim_zeros(ir, 'b')


    def __str__(self):
        return "IR(delay=%d, len(data)=%d)" % (int(self.delay * self.samplerate), len(self.data))

    def __repr__(self):
        return self.__str__()


class Point():

    def __init__(self, x, y=None):
        if y is None and (isinstance(x, list) or isinstance(x, tuple) or isinstance(x, Point)):
            self._x = x[0]
            self._y = x[1]
        else:
            self._x = x
            self._y = y
        #elif radius is not None and direction is not None:
        #    self._x = radius * math.cos(direction)
        #    self._y = radius * math.sin(direction)
        #else:
        #    raise Exception("Error creating point!")

    def polar(direction, radius=1.0):
        return Point(radius * math.cos(direction), radius * math.sin(direction))

    def __hash__(self):
        return hash(self._x) + hash(self._y)

    def __getitem__(self, k):
        if k == 0:
            return self._x
        elif k == 1:
            return self._y
        return (self._x, self._y)[k]

    def get_x(self):
        return self._x
    def set_x(self, v):
        self._x = v

    def get_y(self):
        return self._y
    def set_y(self, v):
        self._y = v

    x = property(get_x, set_x)
    y = property(get_y, set_y)

    def get_radius(self):
        return math.hypot(self._x, self._y)

    def set_radius(self, v):
        new_ratio = v / self.get_radius()
        self._x *= new_ratio
        self._y *= new_ratio

    def get_direction(self):
        return math.atan2(self._y, self._x)
    def set_direction(self, v):
        myr = self.get_radius()
        self._x = myr * math.cos(v)
        self._y = myr * math.sin(v)

    radius = property(get_radius, set_radius)
    direction = property(get_direction, set_direction)
    
    def rotated(self, radians, about=(0.0, 0.0)):
        diff = self - about
        # matrix transformation
        cosp = math.cos(radians)
        sinp = math.sin(radians)
        new_x = cosp * diff._x - sinp * diff._y
        new_y = sinp * diff._x + cosp * diff._y

        return Point(about[0] + new_x, about[1] + new_y)

    def __neg__(self):
        return Point(-self._x, -self._y)

    def __rshift__(self, v):
        # rshift is rotation clockwise (>>)
        return self.rotated(-v)
    
    def __lshift__(self, v):
        # lshift rotation counterclockwise (<<)
        return self.rotated(v)

    def __add__(self, v):
        return Point(self._x + v[0], self._y + v[1])
    def __sub__(self, v):
        return Point(self._x - v[0], self._y - v[1])

    def __dot__(self, v):
        return self._x * v[0] + self._y * v[1]

    def __mul__(self, v):
        if isinstance(v, Point) or isinstance(v, tuple) or isinstance(v, list):
            # dot product
            return self.__dot__(v)
        else:
            return Point(self._x * v, self._y * v)

    def __rmul__(self, v):
        return self.__mul__(v)

    def __div__(self, v):
        if isinstance(v, Point):
            raise Exception("division of points not defined!")
        return Point(self._x / v, self._y / v)

    def __abs__(self):
        return self.r

    def __str__(self):
        return "(%f, %f)" %(self._x, self._y)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, v):
        return self._x == v[0] and self._y == v[1] 


class PointTransformation():

    def __init__(self, translate=None, rotate=None):
        self.translate = translate
        self.rotate = rotate
        if rotate is not None:
            self.rotate_cos = math.cos(rotate)
            self.rotate_sin = math.sin(rotate)

    def identity(self, pt):
        # basecase
        return pt

    def do_translate(self, pt):
        if self.translate is not None:
            return pt

        else:
            return pt + self.translate

    def do_rotate(self, pt):
        if self.rotate is None:
            return pt
        else:
            x = pt[0]
            y = pt[1]
            return Point(x * self.rotate_cos - y * self.rotate_sin, x * self.rotate_sin + y * self.rotate_cos)


    def transform(self, pt):
        if self.rotate is not None:
            x = pt[0] + self.translate[0]
            y = pt[1] + self.translate[1]
            return Point(x * self.rotate_cos - y * self.rotate_sin, x * self.rotate_sin + y * self.rotate_cos)
        else:
            return Point(pt[0] - self.translate[0], pt[1] - self.translate[1])


    def transform_y0(self, _x):
        # transforms assuming pt[1] == 0.0
        if self.rotate is not None:
            x = _x + self.translate[0]
            y = self.translate[1]
            return Point(x * self.rotate_cos - y * self.rotate_sin, x * self.rotate_sin + y * self.rotate_cos)
        else:
            return Point(_x- self.translate[0], - self.translate[1])


    def inverse(self, pt):
        if self.rotate is not None:
            # de-rotate then transform
            # [cos(p) -sin(p)] ^-1
            # [sin(p) cos(p)]
            # = 
            # [cos(p) sin(p)]
            # [-sin(p) cos(p)]
            x = pt[0] * self.rotate_cos + pt[1] * self.rotate_sin
            y = -pt[0] * self.rotate_sin + pt[1] * self.rotate_cos
            return Point(x - self.translate[0], y - self.translate[1])
        else:
            return Point(pt[0] - self.translate[0], pt[1] - self.translate[1])

    def inverse_y0(self, _x):
        # invert assuming pt[1] == 0
        if self.rotate is not None:
            
            # de-rotate then transform
            # [cos(p) -sin(p)] ^-1
            # [sin(p) cos(p)]
            # = 
            # [cos(p) sin(p)]
            # [-sin(p) cos(p)]
            x = _x * self.rotate_cos
            y = -_x * self.rotate_sin
            return Point(x - self.translate[0], y - self.translate[1])
        else:
            return Point(pt[0] - self.translate[0], - self.translate[1])

class Ray():

    def __init__(self, start, direction):
        #start: Point, direction is a radian measure
        self.start = Point(start)
        self.direction = direction

        self._s = None
        self._d = None
        self._T = None

    def perspective_T(self):
        # returns perspective transform
        if self._s is None or self._d is None or self.start != self._s or self.direction != self._d:
            self._s = self.start
            self._d = self.direction
            self._T = PointTransformation(-self.start, -self.direction)
        return self._T


class RaycastResult():

    def __init__(self, hit, point=None, otherside=None, direction=None, distance=None):
        #hit: bool, whether or not the object was hit
        #point: which point did it collide at
        #otherside: where would the ray come through the other side?
        #direction: what is the direction of the tangent line of the geometry at the point of collision?
        #distance: how far away was the hit?

        self.hit = hit
        self.point = point
        self.otherside = otherside
        self.direction = direction
        self.distance = distance

    def __bool__(self):
        return self.hit

    def __str__(self):
        if self.hit:
            return "RaycastResult(True, %s, otherside=%s direction=%f, distance=%f)" % (self.point, self.otherside, self.direction, self.distance)
        else:
            return "RaycastResult(False)"

    def __repr__(self):
        return self.__str__()



class Geometry():
    """

    Geometry is an abstract class that all scene object types (line, circle, etc) interactable inherit from

    All must use the "points" and "values" variables. The points variable includes 2d points that are the pivot points (and can be rotated). The values variables can be anything

    A circle might use self.points = {"center": CenterPoint} and self.values = {"radius": radius}

    Classes can set the default kwargs (for a radius of one, example)

    """

    default_kwargs = {}

    def __init__(self, **kwargs):
        # merge arguments
        self.values = { **self.__class__.default_kwargs, **kwargs }

    def __hash__(self):
        return hash(self.values.values())

    def __getitem__(self, k):
        return self.values.__getitem__(k)

    def __setitem__(self, k, v):
        return self.values.__setitem__(k, v)

    def copy(self):
        return self.__class__(**self.values)

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.values)

    def __repr__(self):
        return self.__str__()

    def __raycast__(self, ray):
        """
        this method should be overloaded and return a RaycastResult detailing how a ray would interact with the geometry
        """
        raise NotImplementedError("__raycast__ method is not implemented")

class Line(Geometry):

    def __init__(self, start, end):
        super().__init__(start=Point(start), end=Point(end))

    def __raycast__(self, ray):
        # transform both points
        #start_p = (self["start"] - ray.start) >> ray.direction
        #end_p = (self["end"] - ray.start) >> ray.direction
        start_p = ray.perspective_T().transform(self["start"])
        end_p = ray.perspective_T().transform(self["end"])

        if (start_p.y >= 0.0 and end_p.y <= 0.0) or (start_p.y <= 0.0 and end_p.y >= 0.0):
            # x point where they intersect
            x_pos = - start_p.y * (end_p.x - start_p.x) / (end_p.y - start_p.y) + start_p.x
            # then we hit the line, but we need to make sure its on the right side (because a ray is only positive)
            x_pos, 
            if x_pos > 0.001:
                hit_pt = ray.perspective_T().inverse_y0(x_pos)
                return RaycastResult(True, hit_pt, hit_pt, (self["end"] - self["start"]).direction, x_pos)
            else:
                return RaycastResult(False)
        else:
            return RaycastResult(False)

class Polygon(Geometry):

    def __init__(self, *points):
        super().__init__(points=points)

        self._h = None
        self._lines = None

    def get_lines(self):
        my_h = hash(self)
        if self._h is None or my_h != self._h:
            self._h = my_h
            if len(self["points"]) == 0 or len(self["points"]) == 1:
                self._lines = []
            elif len(self["points"]) == 2:
                self._lines = [Line(*self["points"])]
            else:
                self._lines = [Line(self["points"][i], self["points"][i + 1]) for i in range(len(self["points"]) - 1)] + [Line(self["points"][-1], self["points"][0])]

        return self._lines

    lines = property(get_lines)

    def __raycast__(self, ray):
        def line_cast(ln):
            # line transform by ray
            #start_p = (ln["start"] - ray.start) >> ray.direction
            #end_p = (ln["end"] - ray.start) >> ray.direction
            return ln.__raycast__(ray)
        
        hits = list(filter(None, map(line_cast, self.lines)))
        if len(hits) == 0:
            return RaycastResult(False)
        elif len(hits) == 1:
            return hits[0]
        else:
            return min(hits, key=lambda x: x.distance)
        

class Circle(Geometry):

    def __init__(self, center, radius=1.0):
        super().__init__(center=Point(center), radius=radius)

    def __raycast__(self, ray):
        # transform both points
        center_p = ray.perspective_T().transform(self["center"])

        r = self["radius"]

        if abs(center_p.y) <= r and center_p.x > 0.001:
            # the third side within the circle
            o_side = math.sqrt(r ** 2 - center_p.y ** 2)
            #dist = center_p.x - o_side
            hit_point = ray.perspective_T().inverse_y0(center_p.x - o_side)
            #other side of the circleo_side
            through_point = ray.perspective_T().inverse_y0(center_p.x + o_side)

            return RaycastResult(True, hit_point, through_point, ray.direction - math.pi/2 + math.atan2(center_p.y, o_side), center_p.x - o_side)
        else:
            return RaycastResult(False)

class SceneObject():
    def __init__(self, tag=None):
        #tag: can be a string
        self.tag = tag

    def __str__(self):
        if self.tag is not None:
            return "%s('%s')" % (self.__class__.__name__, self.tag)
        else:
            return "%s()" % (self.__class__.__name__)

# obj types
class Speaker(SceneObject):

    def __init__(self, geom, tag=None):
        super().__init__(tag)

        #geom: the geometry in the scene
        self.geom = geom

class Material(SceneObject):

    def __init__(self, geom, IR_reflect, IR_through=None, tag=None):
        super().__init__(tag)

        #geom: what geometry is the object in question?
        #IR_reflect: the response from reflections
        #IR_through: the response from the other side
        self.geom = geom
        self.IR_reflect = IR_reflect
        self.IR_through = IR_through


class Mic(SceneObject):
    def __init__(self, pos, tag=None):
        super().__init__(tag)
        #pos: position
        self.pos = Point(pos)



def scene_proc(scene, mic, i, max_bounces, Nsamples):
    cur_dir = 2.0 * math.pi * i / Nsamples
    #st = time.time()
    #_ss_time += time.time() - st

    return scene.single_sample(Ray(mic.pos, cur_dir), max_bounces, anim=None)



class Scene():
    def __init__(self, objs=None, sound_speed=343.0, db_per_doubling=-6.0):
        self.sound_speed = sound_speed
        self.db_per_doubling = db_per_doubling

        if objs is None:
            self.objs = []
        else:
            self.objs = objs

        self.tag_idx = {}

        for i in range(len(self.objs)):
            o = self.objs[i]
            if o.tag != None:
                self.tag_idx[o.tag] = i

    def add_obj(self, obj):
        if obj.tag != None:
            if obj.tag in self.tag_idx.keys():
                print ("warning, duplicate item for tag '%s'" % obj.tag)
            self.tag_idx[obj.tag] = len(self.objs)
        self.objs += [obj]

    def __getitem__(self, k):
        if isinstance(k, int):
            return self.objs[k]
        elif k in self.tag_idx.keys():
            return self.objs[self.tag_idx[k]]
        else:
            raise KeyError()

    def __setitem__(self, k, v):
        if k in self.tag_idx.keys():
            self.objs[self.tag_idx[k]] = v
        else:
            self.add_obj(v)

    def create_IR(self, mic_tag=None, Nsamples=50, max_bounces=25, anim=None, threads=None):
        my_mic = None
        if mic_tag is None:
            for o in self.objs:
                if isinstance(o, Mic):
                    my_mic = o
                    break
        else:
            my_mic = self[mic_tag]
            if my_mic and not isinstance(my_mic, Mic):
                raise Exception("The object under tag '%s' is not a Mic" % (mic_tag))

        total_IR = None

        _ss_time = 0.0

        if threads == 0 or threads == None or threads == 1:
            for i in range(Nsamples):
                if anim:
                    anim.colors["ray"] = (1.0 * i /Nsamples, 1.0 - 1.0 * i / Nsamples, 0.0)
                #print ("%d" % i)
                cur_dir = 2.0 * math.pi * i / Nsamples
                st = time.time()
                cur_IR = self.single_sample(Ray(my_mic.pos, cur_dir), max_bounces, anim)
                _ss_time += time.time() - st
                if total_IR is None:
                    total_IR = cur_IR
                else:
                    total_IR = total_IR + cur_IR
        else:
            # multithreaded approach


            pool = multiprocessing.Pool(threads)
            args = []
            for i in range(Nsamples):
                args += [(self, my_mic, i, max_bounces, Nsamples)]
            coll_IR = pool.starmap(scene_proc, tuple(args))

            # combine
            for c in coll_IR:
                if total_IR is None:
                    total_IR = c
                else:
                    total_IR = total_IR + c



        #print ("individual samples took: %f" % (_ss_time / Nsamples))

        return total_IR


    def single_sample(self, ray, max_bounces=25, anim=None):
        my_ir = None
        total_dist = 0.0

        for i in range(max_bounces):
            hit_obj = None
            bounce_r = RaycastResult(False)
            if anim:
                anim.goto(ray.start)
            
            #print ('bounce: %d' % i)

            for o in self.objs:
                if isinstance(o, Material) or isinstance(o, Speaker):
                    cur_rr = o.geom.__raycast__(ray)

                    if cur_rr:
                        if not bounce_r or cur_rr.distance < bounce_r.distance:
                            bounce_r = cur_rr
                            hit_obj = o

            if hit_obj is not None:
                # a bounce should satisfy "s + s' = 2 * a"
                # so, s' = 2 * a - s
                total_dist += bounce_r.distance

                if anim:
                    anim.draw_ray(ray, distance=bounce_r.distance)

                if isinstance(hit_obj, Speaker):
                    # we found the last
                    break
                else:
                    if my_ir is None:
                        my_ir = hit_obj.IR_reflect.copy()
                    else:
                        # TODO: add IR_through
                        my_ir = my_ir.combine(hit_obj.IR_reflect)

                        #if hit_obj.IR_through is not None:
                        #    my_ir = my_ir.combine(self.single_sample(Ray(bounce_r.otherside, ray.direction), max_bounces=max_bounces // 2, anim=anim))

                    ray = Ray(bounce_r.point, 2 * bounce_r.direction - ray.direction)

            else:
                my_ir = ImpulseResponse.nothing()
                break
        
        if total_dist > 0.0 and my_ir is not None:
            my_ir *= db(math.log(total_dist, 2.0) * self.db_per_doubling)
            my_ir.delay += total_dist / self.sound_speed

        if my_ir is None:
            my_ir = ImpulseResponse.nothing()
        return my_ir
        
 

