import os, sys

def main():
  script_dir = os.path.dirname(os.path.realpath(__file__))

  image_root_dir = os.path.join(script_dir, 'data')
  image_subdirs = [os.path.join(image_root_dir, d) for d in os.listdir(image_root_dir) if os.path.isdir(os.path.join(image_root_dir, d))]

  # Prepare image data for easy loading
  for subdir in image_subdirs:
    print(subdir)
    cmd = f'python3 prepare.py {subdir}'
    print(cmd)
    os.system(cmd)
  
  # Perform classification on image data
  for subdir in image_subdirs:
    name = os.path.basename(subdir)
    csv_file = os.path.join(image_root_dir, f'{name}.csv')

    if not os.path.exists(csv_file):
      print(f'Error: \'{csv_file}\' not found')
      return 1
    
    cmd = f'python3 classify.py {csv_file}'
    print(f'Running \'{cmd}\'')
    os.system(cmd)

  return 0

if __name__ == '__main__':
  exit(main())