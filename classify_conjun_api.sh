#!/bin/zsh

# Directory containing files to process
dir="/Users/raunavghosh/Documents/Arnav Sir's project work/Application of Machine Learning in Detecting Iron Deficiency Anemia Using  Conjunctiva image Dataset from Ghana"

# URL for curl request (replace with your actual URL)
url="13.201.208.34:5000/predict"
# Directory to save files with "true" parameter value
true_dir="conj_detected_anemia"
# Directory to save files with "false" parameter value
false_dir="conj_detected_nonanemia"

# Check if directories exist and create them if not
[[ ! -d "$true_dir" ]] && mkdir -p "$true_dir"
[[ ! -d "$false_dir" ]] && mkdir -p "$false_dir"

# Loop through each file in the directory
for file in "$dir"/*.png; do
  # Check if it's a regular file
  if [[ -f "$file" ]]; then
    # Send file content to the URL using curl
    response=$(curl -X POST -F image=@"$file" "$url")

    # Check if curl was successful (exit code 0)
    if [[ $? -eq 0 ]]; then
      # Parse JSON for the parameter (replace "parameter_name" with the actual name)
      is_true=$(echo "$response" | jq -r '.anemia')

      # Save the file based on the parameter value
      if [[ "$is_true" == "true" ]]; then
        cp "$file" "$true_dir/"
      else
        cp "$file" "$false_dir/"
      fi
    else
      echo "Error sending file: $file (curl exit code: $?)"
    fi
  fi
done
