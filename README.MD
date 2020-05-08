# Simulation of a polarimetric radar in a 3D scene

[![Generic badge](https://img.shields.io/badge/Python-3.6-blue.svg)](https://shields.io/)

### Example:
    import numpy as np
    import matplotlib.pyplot as plt    
    module_location = '../../repos/polarimetric_simulation/'
    sys.path.append(module_location)
    import simulation
    
    sim = mimo.simulation.Simulation()
    
    receiver_name = 'Receiver 1'
    frequencies = np.arange(2.1, 2.9, .01)*1e9
    distances = np.arange(1, 1.3, .1)
    width, height = .4, 1.6
    
    fig, axs = plt.subplots(2, 1)
    for distance in distances:
        sim = simulation.Simulation()
        sim.add_plate(width=width, height=height, position=[distance, 0, 0], normal=[-1, 0, 0], facet_length=facet_length)
        
        sim.add_receiver(name=receiver_name)
        
        for frequency in frequencies:
            sim.add_wave(frequency=frequency, normal=[1, 0, 0])
            
        sim.run()
        sim.plot_receiver(receiver_name, axes=axs)   
    
    _ = plt.legend(np.round(distances,1), title='distance')
    plt.suptitle(f'Svv from plate {round(width,1)}m x {round(height,1)}m')

## Assumptions:

The mathematical representation of the polarimetric radar entities and the physical phenomena are limited as follows:
- Electromagnetic wave has no polarization (to be implemented linear and circular polarizations)
- Surfaces modeling is a series of infinitesimal rigid surfaces (aka: [Facet Method](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003RS002903))
- Facets have 1D reflection gain / radiation pattern (to be implemented 2D)
- No receiver equipment (to be implemented monostatic, bi-static, multi-cells array)
- No transmitter equipment (to be implemented)
- Transmitters have 1D radiation pattern (to be implemented 2D)
- No refraction
- No diffraction
- No resonance
- No atmospheric attenuation
- No Doppler effect
- No interference between close frequencies
- No quantum effect
- Isotropic materials only
- No database of real materials (to be implemented)
- No Montecarlo simulation (to be implemented)
- No plots (to be implemented)