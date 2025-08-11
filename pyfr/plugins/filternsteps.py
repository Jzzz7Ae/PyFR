from pyfr.plugins.base import BaseSolnPlugin

class EntropyFilterPlugin(BaseSolnPlugin):
    name = 'filternsteps'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Read the frequency (apply every N accepted steps)
        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')

    def __call__(self, intg):
        # Apply entropy filter every nsteps accepted steps
        if intg.nacptsteps % self.nsteps == 0:
            if hasattr(intg, 'entropy_filter') and intg.entropy_filter is not None:
                intg.entropy_filter.apply()



