# Assumptions:
# - body is made of solid materials
# - body is a cloud of infinitesimal facets
# - wavelength is smaller than the quantum of the solid surface

# TODO:
# - Polarimetric pulse
# - Polarimetric reflection/scattering
# - Assymetric gain (elevation, azimuth)
# - Receiver
# - Plot received signals
# - Montecarlo with small displacements and attitudes
# - resonance (https://www.radartutorial.eu/17.bauteile/bt47.en.html)
# - difraction
# - refraction
# - Doppler effect
# - Transmitter (equipment with multiple frequencies and polarimetric patterns)


import copy
import numpy as np
import cmath
import math
import itertools
import datetime
import warnings


light_speed = 3e8 # meters / s
    
default = {
    'frequency':       2.45e9,
    'position':        [0, 0, 0],
    'normal':          [1, 0, 0],
    'area':            1,
    'boresight':       [1, 0, 0],
    'level_angle':     0,
    'power':           1,
    'phase':           0,
    'power_cutoff':    1e-9, # (ie: 1e-9 = -90 decibels)
    'gain':            lambda incident_angle: max(0, 1 - incident_angle),
    'name':            None, # receiver name
    'maximum_bounces': 9, # maximum quantity of bounces before quitting the loop
}


def modulus(vector):
    return np.linalg.norm(vector)


def unitary(vector):
    vector_abs = modulus(np.array(vector))
    
    if vector_abs==0:
        warnings.warn("null vector input to unitary function resulted in null vector.")
    
    return np.array(vector) / vector_abs
    

def decibels(power_ratio):
    return 10 * np.log10(power_ratio)
    

def angle(vector_1, vector_2):
    return math.acos(np.clip(np.dot(unitary(vector_1), unitary(vector_2)), -1.0, 1.0))


def cis(phi=0):
    return cmath.exp(1j * phi)


def wave(power, phase):
    return power * cis(phase)


def sphere_area(radius):
    return 4 * np.pi * radius**2


def wavelength(frequency):
    return light_speed / frequency


def propagate(distance, frequency, wave=cis()):
    """Free-space path loss due to lower power density of an isotropic radiator propagating without medium dispersion."""
    density_loss = 1 / sphere_area(distance)
    change_phase = cis(2 * np.pi * frequency / light_speed)
    return wave * density_loss * change_phase


def reflect(incident, normal=[0, 0, 1]):
    incident = unitary(incident)
    normal = unitary(normal)
    reflected_angle = angle(incident, normal)
    reflected_unitary = incident - 2 * np.dot(incident, normal) * normal
    return reflected_angle, reflected_unitary    
    

class Pulse():
    
    def __init__(self, **kwargs):
        self.position = np.array(kwargs['position'])
        self.frequency = kwargs['frequency']
        self.boresight = unitary(np.array(kwargs['boresight']))
        self.gain = kwargs['gain'] # gain_reflection
                     
        if 'wave' in kwargs.keys():
            self.wave = kwargs['wave']
            self.power = modulus(self.wave)
            self.phase = np.arctan2(self.wave.imag, self.wave.real)
        else:
            self.power = kwargs['power']
            self.phase = kwargs['phase']
            self.wave = wave(kwargs['power'], cis(kwargs['phase']))

        
    def __repr__(self):
        return f"<Pulse freq:{self.frequency} wave:{self.wave} position:{self.position}>"
    
    
class Facet():
    
    def __init__(self, **kwargs):
        self.normal = unitary(np.array(kwargs['normal']))
        self.position = np.array(kwargs['position'])
        self.gain = kwargs['gain'] # gain_reflection
        self.name = kwargs['name']
        self.area = kwargs['area']

        
    def __repr__(self):
        return f"<Facet position:{self.position} normal:{self.normal}>"
        
        
class Simulation():
        
    pulses = [] # list of active pulses
    facets = [] # list of infinitesimal surfaces
    receivers = {} # dictionary of receivers
    
    
    def __init__(self):
        self.__dict__ = copy.copy(default)

        
    def __repr__(self):
        pulses = len(self.pulses)
        receivers = len(self.receivers.keys())
        facets = len(self.facets) - receivers
        return f"<Simulation pulses:{pulses} facets:{facets} receivers:{receivers}>"
    
    
    def include_default_parameters(self, **kwargs):
        # copy default parameters
        self.__dict__ = copy.copy(default)
        
        # update parameters with user kwargs
        self.__dict__.update(kwargs)
        
        return self.__dict__
    
    
    def add_pulse(self, **kwargs):
        kwargs = self.include_default_parameters(**kwargs)
            
        # add this new Pulse
        self.pulses.append(Pulse(**kwargs))
        
        
    def add_facet(self, **kwargs):
        kwargs = self.include_default_parameters(**kwargs)
            
        # add this new Facet
        kwargs.update(name=None)
        self.facets.append(Facet(**kwargs))
        
        
    def add_receiver(self, **kwargs):
        kwargs = self.include_default_parameters(**kwargs)
        
        # get name of the receiver
        name = kwargs['name']
        if name is None: name = str(datetime.datetime.now())
            
        # add one Facet at this new receiver position
        kwargs.update(name=name)
        self.facets.append(Facet(**kwargs))
        
        # add receiver to the dictionary member with empty received pulses
        self.receivers[name] = []
        
        
    def run_step(self):
        reflections = []
        for pulse, facet in itertools.product(self.pulses, self.facets):
            
            # geometry calculations
            displacement = facet.position - pulse.position
            distance = modulus(displacement)
    
            if distance > 0:
                boresight_angle = angle(pulse.boresight, displacement)

                reflected_angle, reflected_unitary = reflect(displacement, facet.normal)

                # electromagnetic calculations
                wave_origin = pulse.wave * pulse.gain(boresight_angle) # assuming azimuth==elevation in radiation pattern
                wave_destination = propagate(distance, pulse.frequency, wave_origin)
                reflected_power = wave_destination * facet.area # assuming power constant over all the facet surface. Is it really proportional to the facet area?
#                 print(facet.area)
                print(facet.area, reflected_power, wave_destination)

                is_faded_out = modulus(reflected_power) < default['power_cutoff']

                if not is_faded_out:
                    
                    new_pulse = Pulse(
                            wave=wave_destination,
                            frequency=pulse.frequency, # assuming no Doppler effect
                            position=facet.position,
                            gain=facet.gain,
                            boresight=reflected_unitary,
                        )
                    
                    is_receiver = (facet.name is not None)
                    if is_receiver:
                        
                        # TODO: add receiver gain
                        
                        # add new pulse to the list of this receiver's received pulses
                        self.receivers[facet.name].append(new_pulse)
                    else:
                        # add new pulse as a reflection at this facet's position
                        reflections.append(new_pulse)
                     
        self.pulses = reflections
        
        
    def run(self, **kwargs):
        kwargs = self.include_default_parameters(**kwargs)
        
        maximum_bounces = kwargs['maximum_bounces']
        
        for bounce in range(maximum_bounces):

            self.run_step()

            stop_loop = (len(self.pulses) <= 0)
            if stop_loop:
                break

        if bounce == maximum_bounces - 1:
            warnings.warn("simulation stopped earlier because it reached the maximum bounces.")