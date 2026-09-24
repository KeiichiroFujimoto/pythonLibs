import platform

class osChecker:
   
   @staticmethod
   def getOsType():
      system = platform.system()
      if system == 'Linux':
         return 'Linux'
      elif system == 'Darwin':
         return 'macOS'
      elif system == 'Windows':
         return 'Windows'
      else:
         raise ValueError(f"Unsupported operating system: {system}")



