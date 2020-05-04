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
    'area':            1e-2,
    'boresight':       [1, 0, 0],
    'level_angle':     0,
    'power':           1,
    'phase':           0,
    'power_cutoff':    1e-9, # (ie: 1e-9 = -90 decibels)
    'gain':            lambda incident_angle: max(0, .5*np.cos(5*abs(incident_angle))**.5),
    'name':            None, # receiver name
    'maximum_bounces': 9, # maximum quantity of bounces before quitting the loop
    'receiver':        False, # bool whether it is a facet or a receiver
    'verbose':         False, # print every step of every waves travelling
}

user_parameters = {}


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


def sphere_area(radius):
    return 4 * np.pi * radius**2


def wavelength(frequency):
    return light_speed / frequency


def propagate(distance, frequency, power=1, phase=0):
    """Free-space path loss due to lower power density of an isotropic radiator propagating without medium dispersion."""
    density_loss = 1 / sphere_area(distance)
    phase_change = 2 * np.pi * distance * (frequency / light_speed)
    return power * density_loss, (phase + phase_change) % (2*np.pi)


def reflect(incident, normal=[0, 0, 1]):
    incident = unitary(incident)
    normal = unitary(normal)
    reflected_angle = angle(incident, normal)
    reflected_unitary = incident - 2 * np.dot(incident, normal) * normal
    return reflected_angle, reflected_unitary    
    

class Wave():
    
    def __init__(self, **kwargs):
#         self.position = np.array(kwargs['position'])
#         self.frequency = kwargs['frequency']
#         self.boresight = unitary(np.array(kwargs['boresight']))
#         self.gain = kwargs['gain']
#         self.power = kwargs['power']
#         self.phase = kwargs['phase']
        self.__dict__ = copy.copy(default)
        
        # update parameters with user kwargs
        self.__dict__.update(kwargs)

        
    def __repr__(self):
        return f"<Wave" + \
               f" freq:{round(self.frequency)*1e-9}GHz" + \
               f" power:{round(decibels(self.power))}dbW" + \
               f" \u03C6:{round(self.phase * 180 / np.pi)}\u00B0" + \
               f" position:{self.position}m>"
    
    
class Facet():
    
    def __init__(self, **kwargs):
        self.__dict__ = copy.copy(default)
        
        # update parameters with user kwargs
        self.__dict__.update(kwargs)
        
#         print(kwargs['normal'])
#         self.normal = unitary(np.array(kwargs['normal']))
#         self.position = np.array(kwargs['position'])
#         self.gain = kwargs['gain'] # gain_reflection
#         self.name = kwargs['name']
#         self.area = kwargs['area']
#         self.receiver = kwargs['receiver']

        
    def __repr__(self):
        return f"<Facet position:{self.position} normal:{self.normal}>"
        
        
class Simulation():
        
    waves = [] # list of active waves
    facets = [] # list of infinitesimal surfaces
    receivers = {} # dictionary of receivers
    
    
    def __init__(self, **kwargs):
        self.__dict__ = copy.copy(default)
        
        # update parameters with user kwargs
        self.__dict__.update(kwargs)
        
        self.frequencies = set()

        
    def __repr__(self):
        waves = len(self.waves)
        receivers = len(self.receivers.keys())
        facets = len(self.facets) - receivers
        return f"<Simulation waves:{waves} facets:{facets} receivers:{receivers}>"
    
    
    def include_default_parameters(self, **kwargs):
        # copy default parameters
        user_parameters = copy.copy(default)
        
        # update parameters with user's kwargs
        user_parameters.update(kwargs)
        
        # convert list to np.array
        for key in ['position', 'boresight', 'normal']:
            user_parameters[key] = np.array(user_parameters[key])
        
        return user_parameters
    
    
    def add_wave(self, **kwargs):
        kwargs = self.include_default_parameters(**kwargs)
            
        # add new Wave
        self.waves.append(Wave(**kwargs))
        
        self.frequencies.add(kwargs['frequency'])
        
        
    def add_facet(self, **kwargs):
        kwargs = self.include_default_parameters(**kwargs)
            
        # add new Facet
        kwargs.update(receiver=False)
        self.facets.append(Facet(**kwargs))
        
         
    def add_square_plate(self, **kwargs):
        kwargs = self.include_default_parameters(**kwargs)
        
        n_facets_each_side = 9
        distance_facets = .1
        
        list_facets = range(-n_facets_each_side, n_facets_each_side + 1)
        for x, y in itertools.product([0,], list_facets):
            position = kwargs['position'] + distance_facets * np.array([x, y, 0])
            local_kwargs = copy.copy(kwargs)
            local_kwargs.update(position=position)            
            self.add_facet(**local_kwargs)
        
        
    def add_receiver(self, **kwargs):
        kwargs = self.include_default_parameters(**kwargs)
        
        # get name of the receiver
        name = kwargs['name']
        if name is None: name = str(datetime.datetime.now())
            
        # add one Facet at this new receiver position
        kwargs.update(name=name)
        kwargs.update(receiver=True)
        self.facets.append(Facet(**kwargs))
        
        # add receiver to the dictionary member with empty list of received waves
        self.receivers[name] = []        
        
        
    def sum_waves(self, waves):
        
        power_complex = 0 + 0j
        powers = []
        phases = []

        for wave in waves:
            power_complex += wave.power * simulation.cis(wave.phase)

        power_measured = np.abs(power_complex)
        phase_measured = np.angle(power_complex) % (2*np.pi)
        
        return power_measured, phase_measured
    
    
    def convert_waves_to_measurements(self):            
        # convert list of waves to single measurement for every frequency
        self.frequencies = sorted(list(self.frequencies))
        for receiver_name, receiver_waves in self.receivers.items():
            measurement_per_frequency = {frequency: 0+0j for frequency in self.frequencies}
            for wave in receiver_waves:
                power_complex = wave.power * cis(wave.phase)
                measurement_per_frequency[wave.frequency] += power_complex
                
        self.measurements[receiver_name] = measurement_per_frequency
        
        
    def run_step(self):
        
        reflections = []
        
        for wave, facet in itertools.product(self.waves, self.facets):
            
            # geometry calculations
            displacement = facet.position - wave.position
            distance = modulus(displacement)
    
            if distance > 0: # ignore when facet and wave are at the exact same position

                reflected_angle, reflected_unitary = reflect(displacement, facet.normal)
                reflected_angle = abs(reflected_angle - np.pi)
                
                if self.verbose: print_text = f'Wave freq:{wave.frequency} @ {wave.position}m P={round(decibels(wave.power))}dbW ' + \
                    f'\u03C6={round(wave.phase*180/np.pi)}\u00B0 going to {facet.position}m' + \
                    f' with incidence={round(reflected_angle*180/np.pi)}\u00B0'
                
                is_behind = (np.sin(reflected_angle) <= 0)#(reflected_angle > np.pi/2) and (reflected_angle < 3 * np.pi/2)
                
                if is_behind:
                    
                    if self.verbose: print_text += ' but gone because it approached from behind.'
                    
                else:
                    
                    boresight_angle = angle(wave.boresight, displacement)
                    
                    gain = wave.gain(boresight_angle) # assuming azimuth==elevation in radiation pattern
                    
                    if gain <= 0:
                    
                        if self.verbose: print_text += ' but gone because gain is zero.'
                            
                    else:

                        # electromagnetic calculations
                        power_origin = wave.power * gain
                        power_incident, phase_incident = propagate(distance, wave.frequency, power_origin, wave.phase)
                        phase_incident = phase_incident % (2*np.pi)
                        power_incident *= facet.area # assuming power constant over all the facet surface. Is it really proportional to the facet area?

                        is_faded_out = power_incident < default['power_cutoff']

                        if is_faded_out:

                            if self.verbose: print_text += f' but gone because faded out with P={round(decibels(power_incident))}dbW.'

                        else:

                            new_wave = Wave(
                                    power=power_incident,
                                    phase=phase_incident,
                                    frequency=wave.frequency, # assuming no Doppler effect
                                    position=facet.position,
                                    gain=facet.gain,
                                    boresight=reflected_unitary,
                                )

                            is_receiver = facet.receiver
                            if is_receiver:

                                # TODO: add receiver gain

                                # add new wave to the list of this receiver's received waves
                                self.receivers[facet.name].append(new_wave)
                            else:
                                # add new wave to the list of reflections
                                reflections.append(new_wave)
                                
                
                if self.verbose: print(print_text)
                     
        self.waves = reflections
        
        
    def run(self, **kwargs):
        kwargs = self.include_default_parameters(**kwargs)
        
        maximum_bounces = kwargs['maximum_bounces']
        
        for bounce in range(maximum_bounces):
            
            if self.verbose: print(f'Bounce {bounce} with {len(self.waves)} waves')

            self.run_step()

            stop_loop = (len(self.waves) <= 0)
            if stop_loop:
                break

        if bounce == maximum_bounces - 1:
            warnings.warn("simulation stopped earlier because it reached the maximum bounces.")
            
            
        self.convert_waves_to_measurements()