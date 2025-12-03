for env in $(conda env list | awk '{print $1}' | grep -v "^#" | grep -v "base"); do
  # Remove each environment
  conda env remove --name $env -y
done