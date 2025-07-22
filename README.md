# 🎯 Content Studio - Modern Marketing Content Planner

A beautiful, fully responsive marketing content planner and reviewer with modern bento-style design and intuitive drag-and-drop functionality.

## ✨ Features

### 🎨 Modern Bento Design
- **Flat, modern UI** with clean lines and subtle shadows
- **Bento-style grid layouts** for optimal information density
- **Gradient accents** and sophisticated color schemes
- **Glass morphism effects** with backdrop blur
- **Smooth animations** and micro-interactions

### 📱 Fully Responsive
- **Mobile-first design** with touch-friendly interactions
- **Responsive grid layouts** that adapt to any screen size
- **Mobile sidebar** with slide-out navigation
- **Touch drag-and-drop** support for mobile devices
- **Adaptive typography** and spacing

### 🚀 Built with Modern Tech
- **Next.js 15** with App Router
- **React 19** with latest features
- **shadcn/ui** component library
- **Tailwind CSS v4** for styling
- **TypeScript** for type safety
- **React DnD** for drag-and-drop functionality

## 🎯 Content Planning Features

### 📊 Dashboard Overview
- **Content draft tracking** with beautiful gradient cards
- **Team collaboration** metrics
- **Template library** access
- **Quick action buttons** for settings and save functionality

### 🎪 Drag & Drop Components
- **Media Upload** - Video and image upload components
- **Text Blocks** - Editable text content with live preview
- **Image Gallery** - Multi-image carousel components
- **Form Elements** - Lead capture and conversion forms

### 🎨 Content Types
- **Landing Page Content** - Full landing page builder
- **Meta Form Content** - Social media lead forms
- **Multi-platform support** - Meta, Google, TikTok targeting

### 🛠️ Advanced Features
- **Real-time editing** with inline text editing
- **Component templates** for quick content creation
- **Auto-save functionality** with visual feedback
- **Dark mode support** with system preference detection
- **Team collaboration** for content review

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd content-studio
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🎨 Design System

### Color Palette
The app uses a sophisticated neutral color palette with accent colors:
- **Background**: Clean whites and subtle grays
- **Cards**: Pure white with soft shadows
- **Accents**: Blue primary with purple, green, and orange accents
- **Text**: High contrast dark text with muted secondary text

### Typography
- **Primary Font**: Inter (modern, readable)
- **Headings**: Bold weights with tight tracking
- **Body Text**: Regular weight with optimized line height
- **UI Text**: Medium weight for better hierarchy

### Components
All components use the shadcn/ui design system:
- **Buttons**: Multiple variants (default, outline, ghost)
- **Cards**: Consistent padding and border radius
- **Inputs**: Focus states with ring outlines
- **Badges**: Status indicators with semantic colors

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 768px (single column layout)
- **Tablet**: 768px - 1024px (two column grid)
- **Desktop**: 1024px - 1280px (three column grid)
- **Large Desktop**: > 1280px (four column grid)

### Mobile Features
- **Slide-out sidebar** with sheet component
- **Touch-optimized** drag and drop
- **Simplified navigation** with hamburger menu
- **Stacked layouts** for better mobile UX

## 🔧 Architecture

### Component Structure
```
src/
├── app/                 # Next.js app router
├── components/
│   ├── ui/             # shadcn/ui components
│   ├── Dashboard.tsx   # Main dashboard layout
│   ├── Sidebar.tsx     # Navigation sidebar
│   ├── AdColumn.tsx    # Content builder card
│   └── ...
├── types/              # TypeScript definitions
└── lib/                # Utilities and helpers
```

### State Management
- **React useState** for local component state
- **Props drilling** for simple data flow
- **Context API** ready for global state needs

## 🎯 Content Planning Workflow

1. **Choose Platform** - Select Meta, Google, or TikTok
2. **Pick Content Type** - Landing page or Meta form
3. **Drag Components** - Add media, text, and forms
4. **Edit Content** - Click to edit text inline
5. **Save Changes** - Auto-save with visual feedback
6. **Review & Collaborate** - Share with team for feedback

## 🌙 Dark Mode

The app includes full dark mode support:
- **System preference detection**
- **Manual toggle** (can be added)
- **Consistent dark theme** across all components
- **Proper contrast ratios** for accessibility

## 🔧 Development

### Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

### Adding Components
Use the shadcn/ui CLI to add new components:
```bash
npx shadcn@latest add <component-name>
```

## 🎨 Customization

### Themes
Modify theme variables in `src/app/globals.css`:
```css
:root {
  --background: oklch(1 0 0);
  --foreground: oklch(0.145 0 0);
  /* ... more variables */
}
```

### Components
Extend existing components or create new ones following the design system patterns.

## 📈 Performance

- **Next.js optimization** with automatic code splitting
- **Image optimization** with Next.js Image component
- **CSS optimization** with Tailwind CSS purging
- **Bundle analysis** for size monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

Built with ❤️ using Next.js, React, and shadcn/ui
# reviewapp
