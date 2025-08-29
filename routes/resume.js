const express = require('express');
const router = express.Router();
const Resume = require('../models/Resume'); // MongoDB model

// Save resume data to MongoDB
router.post('/save', async (req, res) => {
  try {
    const resumeData = req.body;
    
    // Create or update resume
    const resume = await Resume.findOneAndUpdate(
      { sessionId: resumeData.sessionId },
      resumeData,
      { upsert: true, new: true }
    );
    
    res.json({ 
      success: true, 
      message: 'Resume saved successfully',
      resumeId: resume._id 
    });
  } catch (error) {
    console.error('Error saving resume:', error);
    res.status(500).json({ error: 'Failed to save resume' });
  }
});

// Get resume data from MongoDB
router.get('/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    const resume = await Resume.findOne({ sessionId });
    
    if (!resume) {
      return res.status(404).json({ error: 'Resume not found' });
    }
    
    res.json(resume);
  } catch (error) {
    console.error('Error fetching resume:', error);
    res.status(500).json({ error: 'Failed to fetch resume' });
  }
});

// Update resume data
router.put('/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const updateData = req.body;
    
    const resume = await Resume.findOneAndUpdate(
      { sessionId },
      updateData,
      { new: true }
    );
    
    if (!resume) {
      return res.status(404).json({ error: 'Resume not found' });
    }
    
    res.json({ 
      success: true, 
      message: 'Resume updated successfully' 
    });
  } catch (error) {
    console.error('Error updating resume:', error);
    res.status(500).json({ error: 'Failed to update resume' });
  }
});

module.exports = router;
