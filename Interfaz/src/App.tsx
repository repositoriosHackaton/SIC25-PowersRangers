import React, { useState, useRef, useEffect } from 'react';
import { BookIcon, Search, X, Send, MessageSquare, Star } from 'lucide-react';
import logo from './logo.png';


// Definición de tipos para los libros
interface Book {
  id: number;
  title: string;
  author: string;
  description: string;
  coverImage: string;
  additionalImages: string[];
  genre: string;  
  pages: number;
  publishYear: number;
  title_es: string;
  description_es: string;
}

// Definición de tipos para los mensajes del chat
interface ChatMessage {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}


// Imágenes adicionales de ejemplo para libros

const sampleAdditionalImages = [
  [
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
  ],
  [
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
  ],
  [
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
  ],
  [
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "hhttps://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
  ],
  [
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "https://images.unsplash.com/photo-1607434472257-d9f8e57a643d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
  ]
];

// Mensajes de ejemplo para el chatbot
const sampleChatMessages: Record<number, ChatMessage[]> = {
  1: [
    {
      id: 1,
      text: "¡Hola! Soy el asistente virtual para 'The Hunger Games'. ¿En qué puedo ayudarte?",
      sender: 'bot',
      timestamp: new Date(Date.now() - 3600000)
    },
    {
      id: 2,
      text: "Me gustaría saber más sobre la trama principal del libro.",
      sender: 'user',
      timestamp: new Date(Date.now() - 3500000)
    },
    {
      id: 3,
      text: "The Hunger Games se desarrolla en un futuro distópico donde el gobierno de Panem organiza un evento anual llamado Los Juegos del Hambre. En estos juegos, jóvenes de cada distrito deben luchar a muerte hasta que solo quede un superviviente. La protagonista, Katniss Everdeen, se ofrece como voluntaria para salvar a su hermana y debe enfrentarse a desafíos mortales mientras descubre una creciente rebelión contra el gobierno.",
      sender: 'bot',
      timestamp: new Date(Date.now() - 3400000)
    }
  ],
  2: [
    {
      id: 1,
      text: "¡Hola! Soy el asistente virtual para 'Shadow's Siege'. ¿En qué puedo ayudarte?",
      sender: 'bot',
      timestamp: new Date(Date.now() - 3600000)
    },
    {
      id: 2,
      text: "¿Quiénes son los personajes principales de este libro?",
      sender: 'user',
      timestamp: new Date(Date.now() - 3500000)
    },
    {
      id: 3,
      text: "Los personajes principales de Shadow's Siege incluyen a Kell, un mago con habilidades únicas; Lira, una guerrera con un pasado misterioso; y Thorne, un estratega brillante con secretos oscuros. Juntos forman un equipo improbable que debe enfrentarse a las fuerzas oscuras que amenazan su mundo.",
      sender: 'bot',
      timestamp: new Date(Date.now() - 3400000)
    }
  ]
};

// URL de la API Flask
const API_URL = 'http://localhost:5000';

function App() {
  const [expandedBookId, setExpandedBookId] = useState<number | null>(null);
  const [recommendations, setRecommendations] = useState<Book[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showRecommendations, setShowRecommendations] = useState(false);
  const [description, setDescription] = useState('');
  const [genre, setGenre] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [expandedImage, setExpandedImage] = useState<string | null>(null);
  const [title, setTitle] = useState('');
  const [activeTab, setActiveTab] = useState<'description' | 'title' | 'generateImages' | 'uploadImage'>('description');
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [ratings, setRatings] = useState<Record<number, number>>({});

  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Cargar mensajes de chat de ejemplo cuando se expande un libro
  useEffect(() => {
    if (expandedBookId !== null) {
      const bookMessages = sampleChatMessages[expandedBookId] || [];
      setChatMessages(bookMessages);
    } else {
      setChatMessages([]);
      setShowChat(false);
    }
  }, [expandedBookId]);

  // Desplazar automáticamente al último mensaje cuando se añade uno nuevo
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
  
    try {
      const response = await fetch(`${API_URL}/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          description: description,
          genre: genre || 'No',
        }),
      });
  
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
  
      const data = await response.json();
  
      const bookRecommendations = data.recommended_books.map((book: { title: string, description: string, author: string, genres: string, pages: string, publishDate: string, cover_image_uri: string, title_es: string, description_es: string }, index: number) => ({
        id: index + 1,
        title: book.title,
        title_es: book.title_es,
        description_es: book.description_es,
        author: book.author,
        description: book.description,
        coverImage: book.cover_image_uri,
        additionalImages: sampleAdditionalImages[index % sampleAdditionalImages.length],
        genre: book.genres,
        pages: book.pages,
        publishYear: book.publishDate,
      }));
  
      setRecommendations(bookRecommendations);
      setShowRecommendations(true);
    } catch (err) {
      console.error('Error al obtener recomendaciones:', err);
      setError('No se pudieron obtener las recomendaciones. Verifica que el servidor Flask esté en ejecución.');
    } finally {
      setIsLoading(false);
    }
  };

const handleTitleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
        // Llamada a la API Flask para obtener recomendaciones por título
        const response = await fetch(`${API_URL}/recommend_by_title`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: title,
                genre: genre || 'No'
            }),
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }

        const data = await response.json();

        // Transformar los títulos de libros en objetos de libro completos
        const bookRecommendations = data.recommended_books.map((book: { title: string, description: string, author: string, genres: string, pages: string, publishDate: string, cover_image_uri: string,title_es: string, description_es:string }, index: number) => ({
            id: index + 1,
            title: book.title,
            title_es: book.title_es,
            description_es: book.description_es,
            author: book.author,
            description: book.description,
            coverImage: book.cover_image_uri,
            additionalImages: sampleAdditionalImages[index % sampleAdditionalImages.length],
            genre: book.genres,
            pages: book.pages,
            publishYear: book.publishDate
        }));

        setRecommendations(bookRecommendations);
        setShowRecommendations(true);
    } catch (err) {
        console.error('Error al obtener recomendaciones:', err);
        setError('No se pudieron obtener las recomendaciones. Verifica que el servidor Flask esté en ejecución.');
    } finally {
        setIsLoading(false);
    }
};

const handleGenerateImages = async (e: React.FormEvent) => {
  e.preventDefault();
  setIsLoading(true);
  setError(null);

  try {
    const imageDescription = (e.target as any).imageDescription.value;

    const response = await fetch(`${API_URL}/generate_images`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        description: imageDescription,
      }),
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }

    const data = await response.json();
    setImageUrls(data.image_urls);  // Actualizar el estado con las URLs de las imágenes
  } catch (err) {
    console.error('Error al generar imágenes:', err);
    setError('No se pudieron generar las imágenes. Verifica que el servidor Flask esté en ejecución.');
  } finally {
    setIsLoading(false);
  }
};

const handleImageUpload = async (e: React.FormEvent) => {
  e.preventDefault();
  setIsLoading(true);
  setError(null);

  if (!selectedImage) {
    setError("Por favor, selecciona una imagen.");
    setIsLoading(false);
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedImage);

  try {
    const response = await fetch(`${API_URL}/recommend_from_image`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json(); // Leer el mensaje de error del backend
      throw new Error(`Error: ${response.status} - ${errorData.error}`);
    }

    const data = await response.json();

    const bookRecommendations = data.recommended_books.map((book: { title: string, description: string, author: string, genres: string, pages: string, publishDate: string, cover_image_uri: string,title_es: string, description_es:string }, index: number) => ({
      id: index + 1,
      title: book.title,
      title_es: book.title_es,
      description_es: book.description_es,
      author: book.author,
      description: book.description,
      coverImage: book.cover_image_uri,
      additionalImages: sampleAdditionalImages[index % sampleAdditionalImages.length],
      genre: book.genres,
      pages: book.pages,
      publishYear: book.publishDate
  }));
    
    setRecommendations(bookRecommendations);
    setShowRecommendations(true);
  } catch (err) {
    console.error("Error al obtener recomendaciones:", err);
    setError("No se pudieron obtener las recomendaciones. Verifica que el servidor Flask esté en ejecución.");
  } finally {
    setIsLoading(false);
  }
};

const generateAdditionalImages = async (book: Book) => {
    try {
        const imageResponse = await fetch(`${API_URL}/generate_images_for_book`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                book_title: book.title
            }),
        });

        if (!imageResponse.ok) {
            throw new Error(`Error: ${imageResponse.status}`);
        }

        const imageData = await imageResponse.json();
        const updatedRecommendations = recommendations.map((b) => 
            b.id === book.id ? { ...b, additionalImages: imageData.image_urls.slice(0, 3) } : b
        );

        setRecommendations(updatedRecommendations);
    } catch (err) {
        console.error('Error al generar imágenes adicionales:', err);
        setError('No se pudieron generar las imágenes adicionales. Verifica que el servidor Flask esté en ejecución.');
    }
};

  const toggleBookExpansion = (bookId: number) => {
    if (expandedBookId === bookId) {
        setExpandedBookId(null);
    } else {
        const book = recommendations.find((b) => b.id === bookId);
        if (book) {
            generateAdditionalImages(book);
        }
        setExpandedBookId(bookId);
    }
};

  const toggleChat = () => {
    if (showChat) {
        // Reiniciar los mensajes del chat y restablecer el estado del personaje actual
        setChatMessages([]);
        setShowChat(false);
    } else {
        setShowChat(true);
        // Iniciar la conversación con el mensaje "¿Con qué personaje desearías hablar?" y la lista de personajes disponibles
        if (expandedBookId !== null) {
            const fetchInitialMessage = async () => {
                try {
                    const bookTitle = recommendations.find(book => book.id === expandedBookId)?.title;

                    const response = await fetch(`${API_URL}/chat`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: '',
                            bookId: expandedBookId,
                            bookTitle: bookTitle // Enviar el título del libro
                        }),
                    });

                    if (!response.ok) {
                        throw new Error(`Error: ${response.status}`);
                    }

                    const data = await response.json();

                    const initialBotMessage: ChatMessage = {
                        id: 0,
                        text: data.response,
                        sender: 'bot',
                        timestamp: new Date()
                    };

                    setChatMessages([initialBotMessage]);
                } catch (err) {
                    console.error('Error al iniciar la conversación:', err);
                }
            };

            fetchInitialMessage();
        }
    }
};

  const handleSendMessage = async () => {
    if (!newMessage.trim() || !expandedBookId) return;
    
    // Añadir mensaje del usuario al chat
    const userMessage: ChatMessage = {
      id: chatMessages.length + 1,
      text: newMessage,
      sender: 'user',
      timestamp: new Date()
    };
    
    setChatMessages([...chatMessages, userMessage]);
    setNewMessage('');
    setIsSendingMessage(true);
    
    try {
      const bookTitle = recommendations.find(book => book.id === expandedBookId)?.title;

      const response = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify({
              message: newMessage,
              bookId: expandedBookId,
              bookTitle: bookTitle // Enviar el título del libro
          }),
      });

      if (!response.ok) {
          throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();

      const botMessage: ChatMessage = {
          id: chatMessages.length + 2,
          text: data.response,
          sender: 'bot',
          timestamp: new Date()
      };

      setChatMessages(prevMessages => [...prevMessages, botMessage]);
    } catch (err) {
      console.error('Error al enviar mensaje:', err);
      // Añadir mensaje de error
      const errorMessage: ChatMessage = {
        id: chatMessages.length + 2,
        text: "Lo siento, ha ocurrido un error al procesar tu mensaje.",
        sender: 'bot',
        timestamp: new Date()
      };
      
      setChatMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsSendingMessage(false);
    }
  };

  // Iniciar la conversación con el mensaje "¿Con qué personaje desearías hablar?" y la lista de personajes disponibles
useEffect(() => {
  if (expandedBookId !== null) {
      const fetchInitialMessage = async () => {
          try {
              const bookTitle = recommendations.find(book => book.id === expandedBookId)?.title;

              const response = await fetch(`${API_URL}/chat`, {
                  method: 'POST',
                  headers: {
                      'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({
                      message: '',
                      bookId: expandedBookId,
                      bookTitle: bookTitle // Enviar el título del libro
                  }),
              });

              if (!response.ok) {
                  throw new Error(`Error: ${response.status}`);
              }

              const data = await response.json();

              const initialBotMessage: ChatMessage = {
                  id: 0,
                  text: data.response,
                  sender: 'bot',
                  timestamp: new Date()
              };

              setChatMessages([initialBotMessage]);
          } catch (err) {
              console.error('Error al iniciar la conversación:', err);
          }
      };

      fetchInitialMessage();
  }
}, [expandedBookId]);


  // Formatear la hora del mensaje
  const formatMessageTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

// Componente de calificación con estrellas
const Rating = ({ initialRating = 0, onRate }: { initialRating?: number; onRate: (rating: number) => void }) => {
  const [rating, setRating] = useState(initialRating);
  const [hoverRating, setHoverRating] = useState(0);

  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          onClick={() => {
            setRating(star);
            onRate(star);
          }}
          onMouseEnter={() => setHoverRating(star)}
          onMouseLeave={() => setHoverRating(0)}
          className="text-yellow-400 hover:text-yellow-500 transition-colors"
        >
          <Star
            size={20}
            fill={
              (hoverRating || rating) >= star
                ? 'currentColor'
                : 'transparent'
            }
          />
        </button>
      ))}
    </div>
  );
};

  // Función para enviar la calificación al backend
  const sendRating = async (rating: number, userInput: string, bookDescription: string) => {
    try {
      console.log('Enviando calificación:', { rating, userInput, bookDescription }); // Depuración
  
      const response = await fetch(`${API_URL}/rate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sentence1: userInput, // Entrada del usuario
          sentence2: bookDescription, // Descripción del libro
          score: rating, // Calificación del usuario (0-5)
        }),
      });
  
      if (!response.ok) {
        throw new Error('Error al enviar la calificación');
      }
  
      console.log('Calificación enviada correctamente');
    } catch (error) {
      console.error('Error:', error);
    }
  };

  // Función para manejar la calificación
  const handleRate = (bookId: number, rating: number) => {
    const book = recommendations.find((book) => book.id === bookId);
    if (!book) {
      console.error('Error: Libro no encontrado');
      return;
    }
  
    // Usar la descripción del libro si la descripción del usuario está vacía
    const userInput = description || book.description_es;
  
    setRatings((prev) => ({ ...prev, [bookId]: rating }));
    sendRating(rating, userInput, book.description_es); // Enviar datos al backend
  };
  
  const Tabs = () => (
    <div className="flex justify-center mb-8">
      <button
        className={`px-4 py-2 ${activeTab === 'description' ? 'bg-primary-100 text-white' : 'bg-white text-primary-100'} rounded-l-lg`}
        onClick={() => setActiveTab('description')}
      >
        Descripción
      </button>
      <button
        className={`px-4 py-2 ${activeTab === 'title' ? 'bg-primary-100 text-white' : 'bg-white text-primary-100'}`}
        onClick={() => setActiveTab('title')}
      >
        Título
      </button>
      <button
        className={`px-4 py-2 ${activeTab === 'generateImages' ? 'bg-primary-100 text-white' : 'bg-white text-primary-100'}`}
        onClick={() => setActiveTab('generateImages')}
      >
        Generar Imágenes
      </button>
      <button
        className={`px-4 py-2 ${activeTab === 'uploadImage' ? 'bg-primary-100 text-white' : 'bg-white text-primary-100'} rounded-r-lg`}
        onClick={() => setActiveTab('uploadImage')}
      >
        Subir Imagen
      </button>
    </div>
  );

  return (
    <div className="min-h-screen bg-bg-100">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <header className="mb-16">
        <div className="max-w-4xl mx-auto flex items-center justify-start gap-3 pl-24">
            <img src={logo} alt="Logo" className="w-40 h-40" />
            <h1 className="text-3xl font-bold text-primary-100">Recomendación de Libros</h1>
        </div>
        </header>

        <main>
          <Tabs />
          {activeTab === 'description' && (
            <form onSubmit={handleSubmit} className="bg-white p-8 rounded-2xl shadow-md max-w-3xl mx-auto mb-16 border border-bg-200">
              <div className="mb-8">
                <label htmlFor="description" className="block font-medium mb-3 text-text-100 text-lg">
                  Describe el tipo de libro que te gustaría leer
                </label>
                <textarea 
                  id="description" 
                  rows={4} 
                  required
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="input-field"
                  placeholder="Por ejemplo: Me gustan los libros de fantasía con elementos de misterio y aventura..."
                />
              </div>

              <div className="mb-8">
                <label htmlFor="genre" className="block font-medium mb-3 text-text-100 text-lg">
                  Género (opcional)
                </label>
                <input 
                  type="text" 
                  id="genre" 
                  value={genre}
                  onChange={(e) => setGenre(e.target.value)}
                  className="input-field"
                  placeholder="Ej: Fantasía, Misterio, Romance..."
                />
              </div>

              <button 
                type="submit" 
                className="w-full p-4 btn-primary flex justify-center items-center"
                disabled={isLoading}
              >
                {!isLoading ? (
                  <span className="flex items-center gap-2">
                    <Search size={20} />
                    <span>Buscar Recomendaciones</span>
                  </span>
                ) : (
                  <span className="w-5 h-5 border-2 border-white rounded-full border-t-transparent animate-spin"></span>
                )}
              </button>
            </form>
          )}

          {activeTab === 'title' && (
            <form onSubmit={handleTitleSubmit} className="bg-white p-8 rounded-2xl shadow-md max-w-3xl mx-auto mb-16 border border-bg-200">
              <div className="mb-8">
                <label htmlFor="title" className="block font-medium mb-3 text-text-100 text-lg">
                  Introduce el título del libro
                </label>
                <input 
                  type="text" 
                  id="title" 
                  required
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="input-field"
                  placeholder="Por ejemplo: Harry Potter y la piedra filosofal"
                />
              </div>

              <div className="mb-8">
                <label htmlFor="genre" className="block font-medium mb-3 text-text-100 text-lg">
                  Género (opcional)
                </label>
                <input 
                  type="text" 
                  id="genre" 
                  value={genre}
                  onChange={(e) => setGenre(e.target.value)}
                  className="input-field"
                  placeholder="Ej: Fantasía, Misterio, Romance..."
                />
              </div>

              <button 
                type="submit" 
                className="w-full p-4 btn-primary flex justify-center items-center"
                disabled={isLoading}
              >
                {!isLoading ? (
                  <span className="flex items-center gap-2">
                    <Search size={20} />
                    <span>Buscar Recomendaciones por Título</span>
                  </span>
                ) : (
                  <span className="w-5 h-5 border-2 border-white rounded-full border-t-transparent animate-spin"></span>
                )}
              </button>
            </form>
          )}

          {activeTab === 'generateImages' && (
            <form onSubmit={handleGenerateImages} className="bg-white p-8 rounded-2xl shadow-md max-w-3xl mx-auto mb-16 border border-bg-200">
              <div className="mb-8">
                <label htmlFor="imageDescription" className="block font-medium mb-3 text-text-100 text-lg">
                  Describe las imágenes que deseas generar
                </label>
                <textarea 
                  id="imageDescription" 
                  rows={4} 
                  required
                  className="input-field"
                  placeholder="Puedes colocar un fragmento del libro para generar tres imágenes..."
                />
              </div>

              <button 
                type="submit" 
                className="w-full p-4 btn-primary flex justify-center items-center"
                disabled={isLoading}
              >
                {!isLoading ? (
                  <span className="flex items-center gap-2">
                    <Search size={20} />
                    <span>Generar Imágenes</span>
                  </span>
                ) : (
                  <span className="w-5 h-5 border-2 border-white rounded-full border-t-transparent animate-spin"></span>
                )}
              </button>

              {/* Mostrar las imágenes generadas */}
              {imageUrls.length > 0 && (
                <div className="mt-8">
                  <h4 className="font-medium mb-4 text-primary-100 text-lg">Imágenes Generadas</h4>
                  <div className="grid grid-cols-2 gap-4">
                    {imageUrls.map((img, index) => (
                      <img
                        key={index}
                        src={img}
                        alt={`Imagen generada ${index + 1}`}
                        className="w-full h-96 object-cover rounded-xl shadow-sm hover:opacity-90 transition-opacity cursor-pointer"  // Cambiado h-40 a h-96
                        onClick={() => setExpandedImage(img)}
                      />
                    ))}
                  </div>
                </div>
              )}
            </form>
          )}

          {activeTab === 'uploadImage' && (
            <form onSubmit={handleImageUpload} className="bg-white p-8 rounded-2xl shadow-md max-w-3xl mx-auto mb-16 border border-bg-200">
              <div className="mb-8">
                <label htmlFor="imageUpload" className="block font-medium mb-3 text-text-100 text-lg">
                  Sube una imagen de uno o varios libros para obtener recomendaciones similares
                </label>
                <input
                  type="file"
                  id="imageUpload"
                  accept="image/*"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setSelectedImage(e.target.files[0]);
                    }
                  }}
                  className="input-field"
                />
              </div>

              <button
                type="submit"
                className="w-full p-4 btn-primary flex justify-center items-center"
                disabled={isLoading}
              >
                {!isLoading ? (
                  <span className="flex items-center gap-2">
                    <Search size={20} />
                    <span>Buscar Recomendaciones</span>
                  </span>
                ) : (
                  <span className="w-5 h-5 border-2 border-white rounded-full border-t-transparent animate-spin"></span>
                )}
              </button>
            </form>
          )}

          {error && (
            <div className="max-w-3xl mx-auto mb-8 p-4 bg-red-100 border border-red-300 rounded-lg text-red-700">
              <p>{error}</p>
              <p className="text-sm mt-2">Asegúrate de que el servidor Flask esté ejecutándose en http://localhost:5000</p>
            </div>
          )}

          {showRecommendations && (
            <div className="max-w-6xl mx-auto">
              <h2 className="text-center text-2xl font-bold mb-12 text-primary-100">Libros Recomendados</h2>
              {recommendations.length === 0 ? (
                <p className="text-center text-text-200">No se encontraron recomendaciones para tu búsqueda. Intenta con otra descripción o género.</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                  {recommendations.map((book) => (
                    <div 
                      key={book.id} 
                      className={`bg-white rounded-2xl overflow-hidden shadow-md border border-bg-200 transition-all duration-300 cursor-pointer ${expandedBookId === book.id ? 'md:col-span-2 lg:col-span-3' : 'hover:shadow-lg hover:-translate-y-1 hover:border-accent-100'}`}
                      onClick={() => toggleBookExpansion(book.id)}
                    >
                      {expandedBookId === book.id ? (
                        <div className="p-8 relative">
                          <div className="flex flex-col lg:flex-row gap-8">
                            <div className="lg:w-1/3">
                              <img 
                                src={book.coverImage} 
                                alt={`Portada de ${book.title_es}`} 
                                className="w-full h-96 object-contain rounded-xl shadow-sm"
                              />
                            </div>
                            <div className="lg:w-2/3">
                              <div className="flex justify-between items-start mb-6">
                                <div>
                                  <h3 className="text-2xl font-bold text-primary-100 mb-1" onClick={(e) => e.stopPropagation()}>{book.title_es}</h3> {/* Cambiado a book.title_es */}
                                  <p className="text-text-200 text-lg" onClick={(e) => e.stopPropagation()}>{book.author}</p>
                                </div>
                                <div className="flex gap-2">
                                  <button 
                                    className="text-text-200 hover:text-accent-100 p-2 rounded-full hover:bg-bg-100 transition-colors"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleChat();
                                    }}
                                    title="Chatear sobre este libro"
                                  >
                                    <MessageSquare size={22} />
                                  </button>
                                  <button 
                                    className="text-text-200 hover:text-accent-100 p-2 rounded-full hover:bg-bg-100 transition-colors"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setExpandedBookId(null);
                                    }}
                                    title="Cerrar"
                                  >
                                    <X size={22} />
                                  </button>
                                </div>
                              </div>
                              <div className="mb-6">
                                <p className="text-text-100 mb-6 leading-relaxed" onClick={(e) => e.stopPropagation()}>{book.description_es}</p> {/* Cambiado a book.description_es */}
                                <div className="grid grid-cols-2 gap-4 text-sm bg-bg-100 p-4 rounded-xl">
                                  <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                                    <span className="font-medium text-primary-100">Género:</span> 
                                    <span>{book.genre}</span>
                                  </div>
                                  <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                                    <span className="font-medium text-primary-100">Páginas:</span> 
                                    <span>{book.pages}</span>
                                  </div>
                                  <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                                    <span className="font-medium text-primary-100">Año:</span> 
                                    <span>{book.publishYear}</span>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                          
                          {/* Sección de imágenes adicionales */}
                          <div className="mt-8">
                            <h4 className="font-medium mb-4 text-primary-100 text-lg">Más imágenes</h4>
                            <div className="grid grid-cols-3 gap-4">
                              {book.additionalImages.map((img, index) => (
                                <img 
                                  key={index} 
                                  src={img} 
                                  alt={`Imagen adicional ${index + 1} de ${book.title_es}`} 
                                  className="w-full h-40 object-cover rounded-xl shadow-sm hover:opacity-90 transition-opacity cursor-pointer"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setExpandedImage(img);
                                  }}
                                />
                              ))}
                            </div>
                          </div>
                          
                          {/* Chatbot */}
                          {showChat && (
                            <div className="fixed bottom-6 right-6 w-80 md:w-96 h-[500px] bg-white rounded-2xl shadow-lg border border-bg-200 flex flex-col z-10 overflow-hidden" onClick={(e) => e.stopPropagation()}>
                              <div className="bg-primary-100 text-white p-4 rounded-t-2xl flex justify-between items-center">
                                <h3 className="font-medium">Chat sobre {book.title_es}</h3> {/* Cambiado a book.title_es */}
                                <button 
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setShowChat(false);
                                  }}
                                  className="text-white hover:text-bg-100 transition-colors"
                                >
                                  <X size={18} />
                                </button>
                              </div>
                              
                              <div 
                                ref={chatContainerRef}
                                className="flex-1 overflow-y-auto p-4 space-y-4"
                                onClick={(e) => e.stopPropagation()}
                              >
                                {chatMessages.map((message) => (
                                  <div 
                                    key={message.id} 
                                    className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                                  >
                                    <div 
                                      className={message.sender === 'user' ? 'chat-bubble-user' : 'chat-bubble-bot'}
                                    >
                                      <p className="text-sm">{message.text}</p>
                                      <span className="text-xs opacity-70 block text-right mt-1">
                                        {formatMessageTime(message.timestamp)}
                                      </span>
                                    </div>
                                  </div>
                                ))}
                                
                                {isSendingMessage && (
                                  <div className="flex justify-start">
                                    <div className="chat-bubble-bot">
                                      <div className="flex space-x-1">
                                        <div className="w-2 h-2 bg-text-200 rounded-full animate-bounce"></div>
                                        <div className="w-2 h-2 bg-text-200 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                                        <div className="w-2 h-2 bg-text-200 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                              
                              <div 
                                className="p-4 border-t border-bg-200 flex gap-2"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <input 
                                  type="text" 
                                  value={newMessage}
                                  onChange={(e) => setNewMessage(e.target.value)}
                                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                                  placeholder="Escribe un mensaje..."
                                  className="flex-1 p-3 border border-bg-300 rounded-xl focus:outline-none focus:border-primary-200"
                                />
                                <button 
                                  onClick={handleSendMessage}
                                  disabled={!newMessage.trim() || isSendingMessage}
                                  className="bg-accent-100 text-white p-3 rounded-xl hover:bg-accent-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                                >
                                  <Send size={20} />
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <>
                          <div className="relative overflow-hidden group">
                            <img 
                              src={book.coverImage} 
                              alt={`Portada de ${book.title_es}`} 
                              className="w-full h-106 object-contain transition-transform duration-500 group-hover:scale-105"
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end">
                              <div className="p-4 text-white">
                                <p className="text-sm font-light">Haz clic para ver detalles</p>
                              </div>
                            </div>
                          </div>
                          <div className="p-5">
                            <h3 className="text-xl font-semibold mb-2 text-primary-100">{book.title_es}</h3> {/* Cambiado a book.title_es */}
                            <p className="text-sm text-text-200 mb-3">{book.author}</p>
                            <p className="text-sm text-text-200 line-clamp-2">{book.description_es}</p> {/* Cambiado a book.description_es */}
                            <div className="mt-3">
                              <Rating
                                initialRating={ratings[book.id] || 0}
                                onRate={(rating) => handleRate(book.id, rating)}
                              />
                              {ratings[book.id] && (
                                <p className="text-sm text-text-200 mt-1">
                                  Calificación: {ratings[book.id]} estrellas
                                </p>
                              )}
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </main>
        
        <footer className="mt-16 text-center text-text-200 text-sm">
          <p>© 2025 Recomendación de Libros | Desarrollado con ❤️</p>
        </footer>
      </div>
      {expandedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex justify-center items-center z-50" onClick={() => setExpandedImage(null)}>
          <img src={expandedImage} alt="Imagen expandida" className="max-w-full max-h-full" />
        </div>
      )}
    </div>
  );
}

export default App;