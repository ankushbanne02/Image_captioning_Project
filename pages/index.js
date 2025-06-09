// pages/index.js
import { useState } from "react";

export default function Home() {
  const [caption, setCaption] = useState("");
  const [imageFile, setImageFile] = useState(null);

  const handleUpload = async () => {
    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64 = reader.result.split(',')[1];
      const res = await fetch('/api/caption', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64 })
      });
      const data = await res.json();
      setCaption(data.caption || data.error);
    };
    reader.readAsDataURL(imageFile);
  };

  return (
    <div>
      <h1>Image Caption Generator</h1>
      <input type="file" accept="image/*" onChange={e => setImageFile(e.target.files[0])} />
      <button onClick={handleUpload}>Generate Caption</button>
      {caption && <h3>Caption: {caption}</h3>}
    </div>
  );
}
