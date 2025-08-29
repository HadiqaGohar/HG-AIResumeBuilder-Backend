const mongoose = require('mongoose');

const resumeSchema = new mongoose.Schema({
  sessionId: {
    type: String,
    required: true,
    unique: true
  },
  name: String,
  tag: String,
  email: String,
  location: String,
  number: String,
  summary: String,
  websites: [String],
  skills: [String],
  education: [String],
  experience: [String],
  student: [String],
  courses: [String],
  internships: [String],
  extracurriculars: [String],
  hobbies: [String],
  references: [String],
  languages: [String],
  templateId: String,
  uploadedFile: {
    filename: String,
    originalName: String,
    size: Number,
    uploadDate: {
      type: Date,
      default: Date.now
    }
  },
  extractedData: mongoose.Schema.Types.Mixed, // For AI extracted data
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Update the updatedAt field before saving
resumeSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

module.exports = mongoose.model('Resume', resumeSchema);
